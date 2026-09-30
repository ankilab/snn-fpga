"""Portable regression entry point. Run from any directory with Python 3.11+."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
os.environ['PYTHONPATH']=str(ROOT/'python')
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def run(command, timeout=180):
    print('+ '+' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),cwd=ROOT,env=os.environ,check=True,timeout=timeout)

def rtl():
    spec=json.loads((ROOT/'configs/rtl_tests.json').read_text())
    compiler=shutil.which('iverilog');vvp=shutil.which('vvp')
    verilator=shutil.which('verilator_bin') or shutil.which('verilator')
    if not all((compiler,vvp,verilator)):
        raise RuntimeError('Install Icarus Verilog (iverilog/vvp) and Verilator, and add them to PATH.')
    common=spec['common_sources'];out=ROOT/'sim';out.mkdir(exist_ok=True)
    for top,source in [('snn_top',common),('snn_axi_lite_wrapper',common+['rtl/snn_axi_lite_wrapper.sv'])]:
        run([verilator,'--lint-only','-Wall','-Wno-PINCONNECTEMPTY','--timing','--top-module',top,*source])
    for case in spec['tests']:
        target=out/(case['name']+'.vvp')
        run([compiler,'-g2012','-s',case['top'],'-o',target,*case['sources']])
        run([vvp,target])
    cases=[('tie',1),('no_output',1),('negative_gap',1),('random',1),('random',7),('random',42)]
    for serial in (False,True):
        target=out/('equivalence_serial.vvp' if serial else 'equivalence.vvp')
        flags=['-Ptb_reference_equivalence.SERIAL_LIF=1'] if serial else []
        run([compiler,'-g2012','-s','tb_reference_equivalence',*flags,'-o',target,*common,'tb/tb_reference_equivalence.sv'])
        for scenario,seed in cases:
            run([sys.executable,'-m','snn_verification.generate_vectors','--output',out/'equivalence','--scenario',scenario,'--seed',seed])
            run([vvp,target,'+CASE='+scenario+'-'+str(seed)])
    return {'directed_rtl_tests':len(spec['tests']),'equivalence_cases':12,'lint_tops':2}

def python_tests(training):
    names=[]
    for p in sorted((ROOT/'python/tests').glob('test_*.py')):
        if p.stem in ('test_torch_training','test_hh_compiled','test_training_observability') and not training:
            continue
        names.append('tests.'+p.stem)
    if training and not importlib.util.find_spec('torch'):
        raise RuntimeError('Training checks require python/requirements-training.txt; use its Python environment.')
    run([sys.executable,'-m','unittest','-v',*names],timeout=600)
    return {'test_modules':len(names),'training_checks_requested':training}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('all','rtl','python','reference'),default='all')
    parser.add_argument('--training',action='store_true',help='Include optional PyTorch/diagnostic/resume tests')
    args=parser.parse_args();start=time.monotonic();result={}
    if args.suite in ('all','rtl'):result['rtl']=rtl()
    if args.suite in ('all','python'):result['python']=python_tests(args.training)
    if args.suite in ('all','reference'):
        run([sys.executable,ROOT/'tools/check_reference.py'])
        result['reference']='passed'
    result.update(status='passed',elapsed_seconds=round(time.monotonic()-start,2),interpreter=sys.version)
    (ROOT/'build').mkdir(exist_ok=True)
    (ROOT/'build'/('checks_'+args.suite+('_training' if args.training else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
