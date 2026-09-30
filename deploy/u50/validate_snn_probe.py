"""Compare a saved U50 probe with the selected reference build and model."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from snn_host.device import HardwareInfo, ModelPackage, SNNDevice

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=('lif','izh','hh'));p.add_argument('probe_json',type=Path)
    a=p.parse_args();ref=json.loads((ROOT/'configs/references.json').read_text())[a.variant]
    values=json.loads(a.probe_json.read_text());build=ref['build']
    expected={name:build[name] for name in ('input_neurons','hidden_neurons','output_neurons','hidden_executor_count','output_executor_count','hidden_neuron_model','output_neuron_model')}
    expected.update(core_id=0x534e4e31,interface_version=0x10000|build['interface_minor'],event_capacity=build['num_events'])
    mismatches={k:{'expected':v,'actual':values.get(k)} for k,v in expected.items() if values.get(k)!=v}
    if mismatches:raise ValueError(json.dumps(mismatches))
    SNNDevice(None).validate_model(ModelPackage.load(ROOT/ref['model_directory']),HardwareInfo(**values))
    print(json.dumps({'status':'passed','variant':a.variant,'scope':'saved probe and model contract; not an inference test'}))
if __name__=='__main__':main()
