#!/usr/bin/env python3
"""Reload selected checkpoints and independently recompute every final metric."""
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def verify_one(path):
    import numpy as np
    import torch as pt
    from model_setup import load_modules
    pt.set_num_threads(1);pt.set_num_interop_threads(1)
    result=json.loads(path.read_text());dataset=result['dataset']
    info=json.loads((HERE/'data'/dataset/'manifest.json').read_text())
    defs,algorithm,convergence=load_modules(result['method'],dataset);module=algorithm.LREANNpt_SUANNmodel
    assert result['data_sha256']==info['data_sha256'] and result['full_source_used']
    assert result['protocol']==json.loads((HERE/'protocol.json').read_text())
    selected=pt.load(path.with_suffix('.best.pt'),map_location='cpu',weights_only=False)
    checkpoint=pt.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
    assert checkpoint['step']==result['steps'] and checkpoint['controller']['stopReason']==result['stop_reason']
    assert result['steps']>=result['convergence_policy']['minimum_updates']
    assert selected['step']==result['selected_step']
    assert selected['train']['loss']==min(c['train']['loss'] for c in result['curves'])
    assert all(pt.equal(v,checkpoint['controller']['best']['model'][k]) for k,v in selected['model'].items())
    config=module.SUANNconfig(128,defs.numberOfLayers,0,defs.hiddenLayerSize,None,info['features'],info['class_count'],1,info['features'],info['class_count'],info['sizes']['train'],None)
    pt.manual_seed(result['seed']);model=module.SUANNmodel(config).cuda()
    assert hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()==result['initial_parameters_sha256']
    model.load_state_dict(selected['model']);model.eval();checks={}
    for split in ['train','validation','test']:
        x=np.load(HERE/'data'/dataset/f'{split}_x.npy',mmap_mode='r');y=np.load(HERE/'data'/dataset/f'{split}_y.npy',mmap_mode='r')
        total=0.;correct=0;counts=np.zeros(info['class_count'],np.int64);hits=np.zeros_like(counts)
        with pt.no_grad():
            for start in range(0,len(y),2048):
                xx=pt.tensor(np.array(x[start:start+2048]),device='cuda');yy=pt.tensor(np.array(y[start:start+2048]),device='cuda')
                model(False,xx,yy,None,None);logits=model.Ztrace[-1]
                total+=pt.nn.functional.cross_entropy(logits,yy,reduction='sum').double().item()
                labels=yy.cpu().numpy();prediction=logits.argmax(1).cpu().numpy();matches=prediction==labels
                correct+=int(matches.sum());counts+=np.bincount(labels,minlength=len(counts));hits+=np.bincount(labels[matches],minlength=len(counts))
        metrics={'rows':len(y),'loss':total/len(y),'accuracy':correct/len(y),'balanced_accuracy':float((hits[counts>0]/counts[counts>0]).mean())}
        for key,value in metrics.items():
            reported=result[split][key]
            if key=='loss':assert abs(value-reported)<=max(2e-6,2e-6*abs(reported)),(split,key,value,reported)
            elif key=='balanced_accuracy':assert abs(value-reported)<1e-8
            else:assert value==reported,(split,key,value,reported)
        checks[split]=metrics
    record={'run':path.stem,'independent_full_split_metrics':checks,'selected_checkpoint_and_stopping_state_verified':True}
    (HERE/'verification'/f'metrics_{path.stem}.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--one',type=Path);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    if args.one:
        print(json.dumps(verify_one(args.one)));return
    protocol=json.loads((HERE/'protocol.json').read_text())
    expected={f'{d}_{m}_seed{s}' for d in protocol['datasets'] for m in protocol['methods'] for s in protocol['seeds']}
    paths=[f for f in (HERE/'runs').glob('*.json') if not f.name.endswith('.progress.json')]
    if args.require_complete:assert {p.stem for p in paths}==expected
    for path in paths:subprocess.run([sys.executable,__file__,'--one',str(path)],check=True)
    #Check paired initialization and minibatch prefix chains across all methods.
    for dataset in protocol['datasets']:
        for seed in protocol['seeds']:
            group=[json.loads(p.read_text()) for p in paths if p.stem.startswith(dataset+'_') and p.stem.endswith(f'_seed{seed}')]
            assert len({r['initial_parameters_sha256'] for r in group})<=1
            chains={}
            for r in group:
                for point in r['curves']:
                    step=point['step']
                    if step in chains:assert chains[step]==point['schedule_chain']
                    else:chains[step]=point['schedule_chain']
    summary={'verified_runs':len(paths),'paired_initialization_and_minibatches_verified':True,'errors':[]}
    (HERE/'verification/saved_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
