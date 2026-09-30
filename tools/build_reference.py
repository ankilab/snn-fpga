"""Build one selected U50 configuration on Linux using its exact saved parameters."""
from pathlib import Path
import argparse
import json
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',choices=('lif','izh','hh'))
    p.add_argument('--mode',choices=('xo','xclbin'),default='xo')
    p.add_argument('--platform',default='')
    p.add_argument('--print-only',action='store_true')
    p.add_argument('--build-dir',type=Path)
    a=p.parse_args()
    compat=os.environ.get('LEGACY_U50_CLOCK_ADAPTER_COMPAT','0')
    if compat not in ('0','1'):p.error('LEGACY_U50_CLOCK_ADAPTER_COMPAT must be 0 or 1')
    if a.mode=='xclbin' and not a.platform:p.error('--platform is required for xclbin')
    spec=json.loads((ROOT/'configs/references.json').read_text())[a.variant]['build']
    env={k.upper():str(spec[k]) for k in ('part','kernel','input_neurons','hidden_neurons','output_neurons','num_events','hidden_executor_count','output_executor_count','hidden_neuron_model','output_neuron_model','interface_minor')}
    env.update(MODE=a.mode,PLATFORM=a.platform,BUILD_DIR=str(a.build_dir.resolve() if a.build_dir else ROOT/'build/reference'/a.variant))
    env['LEGACY_U50_CLOCK_ADAPTER_COMPAT']=compat
    print(json.dumps(env,indent=2),flush=True)
    if not a.print_only:
        if os.name!='posix':p.error('Run the reference build on Linux; see docs/u50.md for PowerShell XO packaging.')
        subprocess.run(['bash',str(ROOT/'deploy/u50/build.sh')],cwd=ROOT,env={**os.environ,**env},check=True)

if __name__=='__main__':main()
