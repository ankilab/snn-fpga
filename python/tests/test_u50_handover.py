"""Integration contracts for the reduced U50 handover, without hardware access."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import io
import json
import tempfile
import unittest
from unittest import mock
from snn_host.cli import _parser, _open_registers, main
from snn_training.campaign_support import _freeze_plan, _source_fingerprint
from snn_training.observability_campaign import build_jobs

ROOT=Path(__file__).resolve().parents[2]

class U50HandoverTests(unittest.TestCase):
    def test_single_xrt_backend_receives_device_kernel_and_path(self):
        args=_parser().parse_args(['--xrt-xclbin','model.xclbin','--xrt-device','2','probe'])
        with mock.patch('snn_host.cli.XrtRegisterIO') as backend:
            _open_registers(args)
            backend.assert_called_once_with(Path('model.xclbin'),device=2,
                kernel='snn_accelerator:{snn_accelerator_1}',server='xrt-register-server')

    def test_qualification_preserves_explicit_horizon_for_empty_events(self):
        reference=mock.Mock();reference.infer.return_value=SimpleNamespace(prediction=None)
        exported=SimpleNamespace(timesteps=8,reference_model=lambda:reference)
        package=SimpleNamespace(exported=exported)
        with mock.patch('snn_host.cli._open_registers'), mock.patch('snn_host.cli.SNNDevice') as device, \
             mock.patch('snn_host.cli.ModelPackage.load',return_value=package), \
             mock.patch('snn_host.cli._upload'), mock.patch('snn_host.cli._events',return_value=()), \
             mock.patch('snn_host.cli.iter_mnist',return_value=iter([(0,bytes(784))])), redirect_stdout(io.StringIO()):
            device.return_value.infer.return_value=None
            result=main(['--xrt-xclbin','model.xclbin','qualify','model','images','labels','--samples','1'])
        self.assertEqual(result,0)
        reference.infer.assert_called_once_with((),timesteps=8)

    def test_50_epoch_matrix_preserves_matched_full_sweep(self):
        matrix=json.loads((ROOT/'configs/full_encoding_50epoch_matrix.json').read_text())
        jobs=build_jobs(matrix,Path('/new-run'),Path('/mnist'))
        self.assertEqual(len(jobs),66)
        self.assertEqual(len({j['id'] for j in jobs}),66)
        self.assertEqual(sum(j['epoch_cap'] for j in jobs),3300)
        signatures=[]
        for model in ('lif','izhikevich','hodgkin_huxley'):
            choices=[c for c in matrix['candidates'] if c['dynamics']==model]
            self.assertEqual(len(choices),22)
            signatures.append({(c['encoding'],json.dumps(c['parameters'],sort_keys=True)) for c in choices})
        self.assertEqual(signatures[0],signatures[1]);self.assertEqual(signatures[1],signatures[2])
        for job in jobs:
            self.assertIsNone(job['parent']);self.assertEqual(job['seed'],14)
            self.assertEqual(job['arguments'][job['arguments'].index('--early-stopping-patience')+1],'0')
            self.assertIn('--training-observability',job['arguments'])

    def test_frozen_plan_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';_freeze_plan(path,{'seed':14})
            _freeze_plan(path,{'seed':14})
            with self.assertRaises(ValueError):_freeze_plan(path,{'seed':15})
            self.assertEqual(json.loads(path.read_text()),{'seed':14})
        self.assertEqual(len(_source_fingerprint(ROOT)),64)

    def test_all_declared_rtl_inputs_exist(self):
        import re
        common=(ROOT/'vivado/common.tcl').read_text()
        names=re.findall(r'\[file join \$::snn_repo_root rtl ([a-z_0-9]+\.sv)\]',common)
        self.assertEqual(len(names),21)
        for name in names:self.assertTrue((ROOT/'rtl'/name).is_file(),name)
        self.assertEqual(len(list((ROOT/'data/hh_luts').glob('*.mem'))),6)

if __name__=='__main__':unittest.main()
