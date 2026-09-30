"""Build an offline viewer; optionally import traces from a standalone HTML."""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent


def save(path, text):
    path.write_text(text, encoding='utf-8', newline='\n')
    return {'path': path.relative_to(HERE).as_posix(), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        help='Import recorded traces from a complete standalone explorer HTML')
    args = parser.parse_args()
    if args.source:
        source = args.source.read_text(encoding='utf-8')
        opening = '<script id="payload" type="application/json">'
        start = source.index(opening) + len(opening)
        payload = source[start:source.index('</script>', start)]
        data = json.loads(payload)
        if data.get('schema') != 'snn-spike-activity-visualization-v1':
            raise ValueError('Unsupported recorded-data schema')
        states = HERE / 'states'
        states.mkdir(exist_ok=True)
        files = []
        for number, trace in enumerate(data['traces']):
            encoded = trace.pop('neuron_states_gzip', None)
            if encoded is not None:
                identifier = f'state-{number:04d}'
                filename = identifier + '.js'
                files.append(save(states / filename,
                                  f'window.neuronStateReceive({json.dumps(identifier)},{json.dumps(encoded)});\n'))
                trace['neuron_states_file'] = filename
        files.append(save(HERE / 'data.js', 'window.neuronExplorerData=' +
                          json.dumps(data, separators=(',', ':'), ensure_ascii=True) + ';\n'))
        if any(f['bytes'] >= 100 * 1024 * 1024 for f in files):
            raise ValueError('An asset exceeds the GitHub file limit')
        save(HERE / 'data_manifest.json', json.dumps({
            'schema': 'neuron-explorer-recorded-data-v1',
            'campaign': data['source_campaign'],
            'traces': len(data['traces']), 'state_traces': len(files) - 1,
            'source_payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'files': files}, indent=2) + '\n')
    if not (HERE / 'data.js').is_file():
        raise ValueError('Missing data.js; supply --source to import recorded data')
    template = (HERE / 'viewer.template.html').read_text(encoding='utf-8')
    if template.count('__NEURON_DATA_SCRIPT__') != 1:
        raise ValueError('Expected exactly one data script placeholder')
    save(HERE / 'index.html', template.replace('__NEURON_DATA_SCRIPT__',
                                             '<script src="data.js"></script>'))
    print('Built explorer/neuron_state/index.html; keep data.js and states/ alongside it.')


if __name__ == '__main__':
    main()
