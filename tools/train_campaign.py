"""Start a new 50-epoch campaign explicitly, without SSH or machine-specific paths."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('prepare','pilot','run'),required=True)
    a=p.parse_args()
    if os.name!='posix':p.error('The GPU campaign scheduler requires Linux. Individual training and CPU tests are portable.')
    output=a.output.resolve()
    evidence=(ROOT/'results').resolve()
    if output==evidence or evidence in output.parents:p.error('Choose a new output directory outside the archived results/')
    env={**os.environ,'PYTHONPATH':str(ROOT/'python')}
    subprocess.run([sys.executable,'-m','snn_training.observability_campaign','--repo-root',str(ROOT),
                    '--root',str(output),'--data-dir',str(a.data_dir.resolve()),
                    '--matrix',str(ROOT/'configs/full_encoding_50epoch_matrix.json'),'--mode',a.mode],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
