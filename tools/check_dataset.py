"""Verify the four uncompressed MNIST IDX files against the campaign hashes."""
from pathlib import Path
import argparse
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw_directory',type=Path)
    a=p.parse_args();expected=json.loads((ROOT/'results/full-encoding-50epoch-2026-09-11/dataset.json').read_text())['sha256']
    for name,digest in expected.items():
        path=a.raw_directory/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Dataset checksum mismatch: '+str(path))
        print(name,'OK')
if __name__=='__main__':main()
