"""Sequential Linux build and U50 qualification with content-bound checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def reusable(record, key):
    return (record.get('status') == 'passed' and record.get('key') == key
            and all(Path(p).is_file() and digest(p) == h
                    for p, h in record.get('outputs', {}).items())
            and bool(record.get('outputs')))


class Run:
    def __init__(self, directory, identity, timeout):
        self.directory = directory
        self.timeout = timeout
        self.path = directory / 'status.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'stages': {}}
        self.identity = identity

    def step(self, name, command, inputs=(), outputs=(), validate=None, reuse=True):
        key = digest_bytes({'identity': self.identity, 'command': [str(x) for x in command],
                            'inputs': {str(p): digest(p) for p in inputs}})
        previous = self.state['stages'].get(name, {})
        if reuse and reusable(previous, key):
            print(f'{name}: reusing verified result', flush=True)
            return
        out = self.directory / (name + '.stdout.log')
        err = self.directory / (name + '.stderr.log')
        record = {'status': 'running', 'key': key, 'started': time.time(),
                  'command': [str(x) for x in command]}
        self.state['stages'][name] = record
        save(self.path, self.state)
        print(f'{name}: starting', flush=True)
        try:
            with out.open('w') as stdout, err.open('w') as stderr:
                process = subprocess.Popen([str(x) for x in command], cwd=ROOT,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    code = process.wait(timeout=self.timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            if code:
                raise RuntimeError(f'exit {code}; see {err}')
            if validate:
                validate(out)
            record.update(status='passed', outputs={str(p): digest(p) for p in [out, err, *outputs]})
        except BaseException as error:
            record.update(status='failed', error=str(error))
            raise
        finally:
            record['finished'] = time.time()
            save(self.path, self.state)


def digest_bytes(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prediction(expected):
    def validate(path):
        result = json.loads(path.read_text())
        if 'prediction' not in result or result['prediction'] != expected:
            raise ValueError(f'Unexpected synthetic prediction: {result}')
    return validate


def qualification(path, samples, gate, expected_accuracy):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != samples + 1:
        raise ValueError('Incomplete sample records')
    rows, summary = records[:-1], records[-1]
    for i, row in enumerate(rows):
        if row.get('index') != i or row.get('label') not in range(10):
            raise ValueError('Invalid sample ordering or label')
        if 'prediction' not in row or row['prediction'] not in [None, *range(10)]:
            raise ValueError('Invalid prediction')
        if gate and ('reference_prediction' not in row or
                     row['prediction'] != row['reference_prediction'] or
                     row.get('matches_reference') is not True):
            raise ValueError('Hardware/reference mismatch')
    accuracy = sum(row['prediction'] == row['label'] for row in rows) / samples
    if summary.get('samples') != samples or summary.get('accuracy') != accuracy:
        raise ValueError('Inconsistent accuracy summary')
    if summary.get('no_prediction') != sum(row['prediction'] is None for row in rows):
        raise ValueError('Inconsistent no-prediction summary')
    if gate:
        if summary.get('reference_mismatches') != 0 or summary.get('reference_accuracy') != accuracy:
            raise ValueError('Reference gate failed')
    elif accuracy != expected_accuracy:
        raise ValueError(f'Accuracy {accuracy} differs from archived {expected_accuracy}; investigate before acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='Four uncompressed MNIST IDX files')
    parser.add_argument('--run-dir', type=Path, required=True, help='Reuse this directory to resume')
    parser.add_argument('--device', type=int, required=True, help='XRT device index reserved for this run')
    parser.add_argument('--gate-samples', type=int, default=10)
    parser.add_argument('--stage-timeout-hours', type=float, default=24)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    compat = os.environ.get('LEGACY_U50_CLOCK_ADAPTER_COMPAT', '0')
    if compat not in ('0', '1'):
        parser.error('LEGACY_U50_CLOCK_ADAPTER_COMPAT must be 0 or 1')
    os.environ['LEGACY_U50_CLOCK_ADAPTER_COMPAT'] = compat
    if sys.platform != 'linux':
        parser.error('Execute on Linux after activating Vivado, Vitis and XRT')
    if not 1 <= args.gate_samples <= 10000 or args.stage_timeout_hours <= 0 or args.device < 0:
        parser.error('Invalid samples, timeout or device')
    platform, data, directory = args.platform.resolve(), args.data_dir.resolve(), args.run_dir.resolve()
    if not platform.is_file() or platform.suffix != '.xpfm':
        parser.error('A readable .xpfm platform is required')
    for tool in ('vivado', 'v++', 'g++', 'bash', 'xrt-smi'):
        if not shutil.which(tool):
            parser.error(f'Missing tool: {tool}')
    os.environ['PYTHONPATH'] = str(ROOT / 'python')
    os.environ['PYTHONUNBUFFERED'] = '1'
    for script, extra in [('check_dataset.py', [data]), ('check_reference.py', [])]:
        subprocess.run([sys.executable, ROOT / 'tools' / script, *extra], check=True)
    # Hash the source inputs, not generated build products or old Git history.
    sources = [*(ROOT / 'rtl').glob('*.sv'), *ROOT.glob('examples/*.raw')]
    refs = json.loads((ROOT / 'configs/references.json').read_text())
    sources.extend(ROOT / f['path'] for ref in refs.values() for f in ref['model_files'])
    for folder in ('python', 'vivado', 'deploy', 'tools', 'configs', 'data/hh_luts'):
        sources.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                       and '__pycache__' not in p.parts and p.suffix != '.pyc')
    identity = {'source': {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
                'platform': str(platform), 'platform_sha256': digest(platform),
                'platform_files': {str(p.relative_to(platform.parent)): digest(p)
                                   for p in sorted(platform.parent.rglob('*')) if p.is_file()},
                'device': args.device, 'gate_samples': args.gate_samples,
                'legacy_u50_clock_adapter_compat': compat,
                'data': {p.name: digest(p) for p in data.iterdir() if p.name.startswith('t10k-')},
                'tools': {t: subprocess.check_output([t, '-version' if t == 'vivado' else '--version'],
                           stderr=subprocess.STDOUT, timeout=60).decode(errors='replace')
                          for t in ('vivado', 'v++', 'xrt-smi')}}
    print('Preflight passed', flush=True)
    if args.preflight_only:
        return 0
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    import fcntl
    # Per-user, per-device lock also covers different run directories/checkouts.
    lockdir = Path.home() / '.cache/snn-u50'
    lockdir.mkdir(parents=True, exist_ok=True)
    lock = (lockdir / f'device-{args.device}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Inputs/tool versions changed. Use a new run directory.')
    save(identity_path, identity)
    common = Run(directory, identity, args.stage_timeout_hours * 3600)
    common.step('inventory', ['xrt-smi', '--batch', 'examine'], reuse=False)
    server = directory / 'xrt-register-server'
    common.step('host_build', ['bash', ROOT / 'deploy/u50/build_host.sh', server], outputs=[server])
    refs = json.loads((ROOT / 'configs/references.json').read_text())
    summary = {}
    for variant, ref in refs.items():
        work = directory / variant
        work.mkdir(exist_ok=True)
        run = Run(work, identity, args.stage_timeout_hours * 3600)
        try:
            # Each failed/interrupted build gets a fresh directory; no stale Vitis products.
            old = run.state['stages'].get('build', {})
            build = Path(old.get('command', [''])[-1]) if reusable(old, old.get('key')) else work / f'build-{time.time_ns()}'
            xclbin = build / 'artifacts/snn_accelerator.xclbin'
            run.step('build', [sys.executable, ROOT / 'tools/build_reference.py', variant,
                              '--mode', 'xclbin', '--platform', platform, '--build-dir', build], outputs=[xclbin])
            checkpoints = list(build.rglob('*_routed.dcp'))
            if len(checkpoints) != 1:
                raise ValueError(f'Expected one routed checkpoint, found {len(checkpoints)}; inspect build reports')
            route, timing = work / 'route.rpt', work / 'timing.rpt'
            run.step('timing', ['vivado', '-mode', 'batch', '-notrace', '-source',
                               ROOT / 'deploy/u50/check_routed.tcl', '-tclargs', checkpoints[0], route, timing],
                     inputs=[xclbin, checkpoints[0]], outputs=[route, timing])
            base = [sys.executable, '-m', 'snn_host', '--xrt-xclbin', xclbin,
                    '--xrt-device', str(args.device), '--xrt-server', server, '--timeout', '60']
            model = ROOT / ref['model_directory']
            inputs = [xclbin, server, *(ROOT / f['path'] for f in ref['model_files'])]
            # Probe always reloads and checks the physical device on resume.
            run.step('probe', [*base, 'probe'], inputs=inputs, reuse=False)
            run.step('probe_validation', [sys.executable, ROOT / 'deploy/u50/validate_snn_probe.py',
                                         variant, work / 'probe.stdout.log'], inputs=[work / 'probe.stdout.log'], reuse=False)
            for pattern, expected in [('blank', None), ('vertical_bar', 1)]:
                run.step(pattern, [*base, 'infer', model, ROOT / f'examples/{pattern}.raw'],
                         inputs=inputs, validate=prediction(expected))
            for stage, command, count in [('gate', 'qualify', args.gate_samples), ('full', 'mnist', 10000)]:
                run.step(stage, [*base, command, model, data / 't10k-images-idx3-ubyte',
                                data / 't10k-labels-idx1-ubyte', '--samples', str(count)], inputs=inputs,
                         validate=lambda p, n=count, gate=stage == 'gate': qualification(
                             p, n, gate, ref['test_accuracy_reported']))
            summary[variant] = {'status': 'passed', 'xclbin_sha256': digest(xclbin),
                                'model_files': ref['model_files']}
        except Exception as error:
            summary[variant] = {'status': 'failed', 'error': str(error)}
            print(f'{variant}: FAILED: {error}', flush=True)
        save(directory / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(any(v['status'] != 'passed' for v in summary.values()))


if __name__ == '__main__':
    raise SystemExit(main())
