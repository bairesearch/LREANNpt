#!/usr/bin/env python3
"""Load every source row, partition without subsampling, fit preprocessing on train."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from reproduction_support import (hub_source, bundled_source, download_verified, expected_manifest,
    prepared_data_matches, check_prepared_manifest, ensure_directories)
from sklearn.model_selection import train_test_split

HERE=Path(__file__).resolve().parent
OUT=HERE/'data'; SEED=20260930
SOURCES={
 'tabular-benchmark':('inria-soda/tabular-benchmark','class',['clf_cat/albert.csv']),
 'blog-feedback':('wwydmanski/blog-feedback','target',['train.csv','test.csv']),
 'red-wine':('lvwerra/red-wine','quality',['winequality-red.csv']),
 'breast-cancer-wisconsin':('scikit-learn/breast-cancer-wisconsin','diagnosis',['breast_cancer.csv']),
 'diabetes-readmission':('imodels/diabetes-readmission','readmitted',['train.csv','test.csv']),
 'banking-marketing':('Andyrasika/banking-marketing','y',['data/train-00000-of-00001.parquet','data/test-00000-of-00001.parquet']),
 'adult_income_dataset':('meghana/adult_income_dataset','income',['adult.csv']),
 'iris':('scikit-learn/iris','Species',['Iris.csv']),
}

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def source_info(path,url):
    return {'url':url,'path':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)}

def hub(repo,filename):
    return hub_source(repo, filename)

def split_indices(indices,y,fraction):
    _,counts=np.unique(y[indices],return_counts=True)
    stratify=y[indices] if counts.min()>=2 and len(counts)<=len(indices)*min(fraction,1-fraction) else None
    return train_test_split(indices,test_size=fraction,stratify=stratify,random_state=SEED)

def partitions(y,split=None):
    if split is None:
        train,heldout=split_indices(np.arange(len(y)),y,.4)
        validation,test=split_indices(heldout,y,.5)
    else:
        train=np.flatnonzero(split=='train');validation=np.flatnonzero(split=='validation');test=np.flatnonzero(split=='test')
        if not len(validation):train,validation=split_indices(train,y,.2)
    result={'train':train,'validation':validation,'test':test}
    assert all(len(v)>0 for v in result.values())
    assert np.array_equal(np.sort(np.concatenate(list(result.values()))),np.arange(len(y))), 'Every source row must belong to exactly one split'
    return result

def finish(name,raw,y,parts,features,sources,excluded=None,categories=None,notes=None):
    directory=OUT/name;directory.mkdir(parents=True,exist_ok=True)
    n,f=raw.shape
    assert sum(map(len,parts.values()))==n and len(y)==n
    # Min/max computed over every training row, in bounded memory; no sampling.
    minimum=np.full(f,np.inf,dtype=np.float32);maximum=np.full(f,-np.inf,dtype=np.float32)
    for start in range(0,len(parts['train']),8192):
        values=np.asarray(raw[parts['train'][start:start+8192]],dtype=np.float32)
        assert np.isfinite(values).all()
        minimum=np.minimum(minimum,values.min(0));maximum=np.maximum(maximum,values.max(0))
    scales=np.array([0. if float(hi)==float(lo) else 1./(float(hi)-float(lo)+1e-8) for lo,hi in zip(minimum,maximum)],dtype=np.float32)
    effective=np.where(scales==0,np.float32(1),scales)
    stats={str(c):['minmax',float(a),0. if float(b)==float(a) else 1./(float(b)-float(a)+1e-8)] for c,a,b in zip(features,minimum,maximum)}
    files={}
    for split,indices in parts.items():
        out=np.lib.format.open_memmap(directory/f'{split}_x.npy',mode='w+',dtype=np.float32,shape=(len(indices),f))
        labels=np.lib.format.open_memmap(directory/f'{split}_y.npy',mode='w+',dtype=np.int64,shape=(len(indices),))
        for start in range(0,len(indices),8192):
            idx=indices[start:start+8192]
            out[start:start+len(idx)]=(np.asarray(raw[idx],dtype=np.float32)-minimum)*effective
            labels[start:start+len(idx)]=y[idx]
            assert np.isfinite(out[start:start+len(idx)]).all()
        out.flush();labels.flush();del out,labels
        np.save(directory/f'{split}_source_indices.npy',indices)
        for suffix in ['x','y','source_indices']:
            path=directory/f'{split}_{suffix}.npy'
            files[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    info={'dataset':name,'source_rows_loaded':n,'all_source_rows_used':True,'source_prefix_used':False,'row_caps':None,
          'sizes':{s:len(idx) for s,idx in parts.items()},'features':f,'feature_names':features,'class_count':int(np.max(y))+1,
          'split_seed':SEED,'sources':sources,'dropped_columns':excluded or [],'category_mappings':categories or {},
          'normalisation_statistics':stats,'normalisation_fit_rows':len(parts['train']),'normalisation_fit_split':'train',
          'preprocessing':'Train-fitted category maps and float32 min/max equations of patched ANNpt_data; streamed without row limits. Verified separately against production normalization.',
          'files':files,'notes':notes or [],'partition_coverage_verified':True}
    info['data_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    check_prepared_manifest(name, info)
    atomic_json(directory/'manifest.json',info)
    print('PREPARED',name,json.dumps(info['sizes']),f,'features',flush=True)
    return info

def store_frame(name,frame,label,split,sources,notes=None):
    frame=frame.reset_index(drop=True)
    target=frame[label]
    if name=='blog-feedback':y=target.to_numpy(dtype=np.int64)
    else:
        classes=sorted(target.dropna().unique().tolist(),key=str)
        y=target.map({v:i for i,v in enumerate(classes)}).to_numpy(dtype=np.int64)
    assert not target.isna().any()
    parts=partitions(y,split)
    excluded=[c for c in frame.columns if str(c).lower() in ('id','patient_nbr','encounter_id') or str(c).startswith('Unnamed:')]
    features=[c for c in frame.columns if c not in excluded and c!=label]
    empty=[c for c in features if frame[c].iloc[parts['train']].isna().all()]
    excluded+=empty;features=[c for c in features if c not in empty]
    directory=OUT/name;directory.mkdir(parents=True,exist_ok=True)
    raw=np.lib.format.open_memmap(directory/'raw_x.npy',mode='w+',dtype=np.float32,shape=(len(frame),len(features)))
    categories={}
    for j,col in enumerate(features):
        series=frame[col]
        if series.dtype=='object' or isinstance(series.dtype,pd.CategoricalDtype):
            values=series.fillna('__missing__').astype(str).str.strip()
            coding={v:i for i,v in enumerate(sorted(values.iloc[parts['train']].unique()))}
            raw[:,j]=values.map(coding).fillna(-1).to_numpy(dtype=np.float32);categories[str(col)]=coding
        else:raw[:,j]=pd.to_numeric(series,errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        raw[:,j]=np.nan_to_num(raw[:,j],nan=0,posinf=0,neginf=0)
    raw.flush()
    finish(name,raw,y,parts,list(map(str,features)),sources,excluded,categories,notes)
    del raw
    (directory/'raw_x.npy').unlink()

def prepare_hf(name):
    repo,label,names=SOURCES[name];frames=[];splits=[];sources=[]
    for filename in names:
        path=hub(repo,filename)
        frame=pd.read_parquet(path) if filename.endswith('.parquet') else pd.read_csv(path)
        frames.append(frame);splits.extend(['test' if 'test' in filename else 'train']*len(frame))
        sources.append(source_info(path,f'https://huggingface.co/datasets/{repo}/resolve/{path.parts[-len(Path(filename).parts)-1]}/{filename}'))
    store_frame(name,pd.concat(frames,ignore_index=True),label,np.asarray(splits) if len(names)>1 else None,sources,
        ['Integer comment counts treated as class IDs, following repository task convention.'] if name=='blog-feedback' else None)

def prepare_special(name):
    if name=='new-thyroid':
        path=bundled_source(name, 'new-thyroid.csv')
        store_frame(name,pd.read_csv(path),'class',None,[source_info(path,'Repository data/new-thyroid.csv')])
    elif name=='titanic':
        path=bundled_source(name, 'titanic_openml_40945.csv')
        frame=pd.read_csv(path)[['pclass','sex','age','sibsp','parch','fare','embarked','survived']]
        frame['sex']=frame.sex.map({'male':0.,'female':1.})
        frame['embarked']=frame.embarked.map({'S':0.,'C':1.,'Q':2.})
        frame=frame.fillna(0)
        store_frame(name,frame,'survived',None,[source_info(path,'https://www.openml.org/d/40945')])
    elif name=='covertype':
        from sklearn.datasets import fetch_covtype
        data=fetch_covtype(data_home=str(HERE/'sources/sklearn'),download_if_missing=True)
        frame=pd.DataFrame(data.data,columns=data.feature_names);frame['cover_type']=data.target
        store_frame(name,frame,'cover_type',None,[{'url':'https://archive.ics.uci.edu/dataset/31/covertype','loader':'sklearn.fetch_covtype','rows':len(frame)}])
    elif name=='topquark':
        import pyarrow.parquet as pq
        directory=OUT/name;directory.mkdir(parents=True,exist_ok=True)
        paths={s:(hub('lewtun/top_quark_tagging',f'data/{s}-raw.parquet'),hub('lewtun/top_quark_tagging',f'data/{s}-labels.parquet')) for s in ['train','validation','test']}
        sizes={s:pq.ParquetFile(p[0]).metadata.num_rows for s,p in paths.items()}
        names=pq.ParquetFile(paths['train'][0]).schema_arrow.names
        excluded=[c for c in names if c.lower()=='ttv' or any(t in c.lower() for t in ['truth','is_signal','label','target'])]
        features=[c for c in names if c not in excluded]
        n=sum(sizes.values())
        raw=np.lib.format.open_memmap(directory/'raw_x.npy',mode='w+',dtype=np.float32,shape=(n,len(features)))
        y=np.empty(n,dtype=np.int64);parts={};sources=[];offset=0
        for split,(rawpath,labelpath) in paths.items():
            labels=pd.read_parquet(labelpath)['is_signal_new'].to_numpy(dtype=np.int64)
            assert len(labels)==sizes[split]
            start=offset
            for batch in pq.ParquetFile(rawpath).iter_batches(batch_size=8192,columns=features):
                array=batch.to_pandas().to_numpy(dtype=np.float32)
                assert np.isfinite(array).all()
                raw[offset:offset+len(array)]=array;offset+=len(array)
            assert offset-start==sizes[split]
            y[start:offset]=labels;parts[split]=np.arange(start,offset)
            sources.extend([source_info(p,f'https://huggingface.co/datasets/lewtun/top_quark_tagging/resolve/{p.parent.parent.name}/data/{p.name}') for p in [rawpath,labelpath]])
            print('READ FULL TOPQUARK',split,sizes[split],flush=True)
        raw.flush()
        finish(name,raw,y,parts,features,sources,excluded,notes=['Every row of each official raw/label split; truth and ttv predictors excluded.'])
        del raw;(directory/'raw_x.npy').unlink()
    elif name=='higgs':
        directory=OUT/name;directory.mkdir(parents=True,exist_ok=True)
        source=expected_manifest(name)['sources'][0]
        path=download_verified(source['url'], HERE/'sources/HIGGS.csv.gz', source)
        n=11000000
        raw=np.lib.format.open_memmap(directory/'raw_x.npy',mode='w+',dtype=np.float32,shape=(n,28))
        y=np.empty(n,dtype=np.int64);offset=0
        for frame in pd.read_csv(path,header=None,dtype=np.float32,chunksize=100000):
            end=offset+len(frame)
            if end>n:raise ValueError('Unexpected additional HIGGS rows; verify source')
            raw[offset:end]=frame.iloc[:,1:].to_numpy();y[offset:end]=frame.iloc[:,0].to_numpy(dtype=np.int64);offset=end
            if offset%1000000==0:print('READ HIGGS',offset,flush=True)
        assert offset==n,'Incomplete HIGGS source'
        raw.flush()
        finish(name,raw,y,partitions(y),[f'feature_{i}' for i in range(28)],
            [source_info(path,'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')],
            notes=['All 11,000,000 rows. Retains prior random 60/20/20 partition rule; this is not the original HIGGS paper final-500,000 test partition.'])
        del raw;(directory/'raw_x.npy').unlink()
    else:raise ValueError(name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=json.loads((HERE/'protocol.json').read_text())['datasets']);args=parser.parse_args()
    ensure_directories()
    for name in args.datasets:
        if prepared_data_matches(name):
            print('VERIFIED EXISTING DATA', name, flush=True)
            continue
        atomic_json(HERE/'PREPARATION.json',{'status':'preparing','dataset':name,'time':time.time()})
        try:
            if name in SOURCES:prepare_hf(name)
            else:prepare_special(name)
        except Exception:
            atomic_json(HERE/'logs'/f'preparation_failure_{name}.json',{'dataset':name,'traceback':traceback.format_exc()})
            raise
    atomic_json(HERE/'PREPARATION.json',{'status':'complete_for_requested_datasets','datasets':args.datasets,'time':time.time()})

if __name__=='__main__':main()
