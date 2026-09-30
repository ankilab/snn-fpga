"""Check the released model bytes and the two synthetic inputs, without FPGA access."""
from pathlib import Path
import dataclasses
import hashlib
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from snn_host.device import ModelPackage, HardwareInfo, SNNDevice

def main():
    refs=json.loads((ROOT/'configs/references.json').read_text())
    examples=json.loads((ROOT/'examples/reference_examples.json').read_text())
    for variant,ref in refs.items():
        for f in ref['model_files']:
            if hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()!=f['sha256']:
                raise ValueError('Model bytes changed: '+f['path'])
        package=ModelPackage.load(ROOT/ref['model_directory'])
        probe=json.loads((ROOT/f'evidence/u50-reference-2026-09-23/{variant}/probe.json').read_text())
        # A missing IO backend makes accidental hardware access fail immediately.
        SNNDevice(None).validate_model(package,HardwareInfo(**probe))
        for pattern in examples['patterns']:
            raw=(ROOT/pattern['raw_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pattern['sha256']
            events=package.exported.encode_events(list(raw),sample_index=pattern['sample_index'])
            result=package.exported.reference_model().infer(events,timesteps=8)
            expected=next(x for x in examples['expected'] if x['variant']==variant and x['input']==pattern['id'])
            actual=json.loads(json.dumps(dataclasses.asdict(result)))
            assert len(events)==expected['input_events']
            assert actual==expected['reference_result'],(variant,pattern['id'])
            print(variant,pattern['id'],'prediction=',result.prediction,'class_counts=',result.class_counts)
    print('Three packages, recorded parameter contracts and six synthetic reference cases passed. No hardware test.')

if __name__=='__main__':main()
