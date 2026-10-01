"""Independently verify uncapped row coverage, disjoint source indices, files and schema."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+');args=p.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text());checks=[]
    for name in args.datasets or protocol['datasets']:
        directory=HERE/'data'/name;info=json.loads((directory/'manifest.json').read_text())
        assert info['row_caps'] is None and info['all_source_rows_used'] and not info['source_prefix_used']
        assert sum(info['sizes'].values())==info['source_rows_loaded']
        coverage=np.zeros(info['source_rows_loaded'],dtype=np.uint8)
        for split,n in info['sizes'].items():
            indices=np.load(directory/f'{split}_source_indices.npy',mmap_mode='r')
            assert len(indices)==n and len(np.unique(indices))==n
            assert indices.min()>=0 and indices.max()<len(coverage)
            coverage[indices]+=1
            x=np.load(directory/f'{split}_x.npy',mmap_mode='r');y=np.load(directory/f'{split}_y.npy',mmap_mode='r')
            assert x.shape==(n,info['features']) and y.shape==(n,)
            for start in range(0,n,8192):
                assert np.isfinite(x[start:start+8192]).all()
                assert (y[start:start+8192]>=0).all() and (y[start:start+8192]<info['class_count']).all()
        assert (coverage==1).all(), 'Lost or overlapping source rows'
        assert info['normalisation_fit_rows']==info['sizes']['train']
        for name_file,meta in info['files'].items():
            path=directory/name_file
            assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256']
        checks.append({'dataset':name,'source_rows':len(coverage),'splits':info['sizes'],'every_source_row_used_exactly_once':True,'prepared_file_hashes_verified':True})
        print('AUDITED',checks[-1],flush=True)
    record={'time':time.time(),'checks':checks,'errors':[]}
    (HERE/'verification/full_data_audit.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
