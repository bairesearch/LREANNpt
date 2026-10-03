#!/usr/bin/env python3
"""Fresh uncapped SUANN training; shared training- or validation-loss controller."""
import argparse, concurrent.futures, hashlib, json, math, multiprocessing as mp, os, shutil, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
RUNS=HERE/'runs';WORK=HERE/'checkpoints'
from reproduction_support import require_workspace

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run_one(job):
    require_workspace()
    dataset,method,seed=job
    import numpy as np
    import torch as pt
    from model_setup import load_modules
    from cached_graphed_population import CachedGraphedPopulationStep
    pt.set_num_threads(1);pt.set_num_interop_threads(1)
    directory=HERE/'data'/dataset
    info=json.loads((directory/'manifest.json').read_text())
    assert info['all_source_rows_used'] and info['partition_coverage_verified'] and info['row_caps'] is None
    assert sum(info['sizes'].values())==info['source_rows_loaded']
    digest=hashlib.sha256(json.dumps(info['files'],sort_keys=True).encode()).hexdigest()
    assert digest==info['data_sha256']
    #Source is immutable for this experiment; fail rather than silently change it.
    for name,expected in PROTOCOL['source_hashes'].items():
        assert hashlib.sha256((HERE/'source/LREANNpt'/name).read_bytes()).hexdigest()==expected
    tag=f'{dataset}_{method}_seed{seed}';output=RUNS/f'{tag}.json'
    if output.exists():
        result=json.loads(output.read_text())
        assert result['status']=='complete' and result['data_sha256']==digest and result['protocol']==PROTOCOL
        return result
    definitions,algorithm,convergence=load_modules(method,dataset)
    module=algorithm.LREANNpt_SUANNmodel;device=pt.device('cuda')
    arrays={s:(np.load(directory/f'{s}_x.npy',mmap_mode='r'),np.load(directory/f'{s}_y.npy',mmap_mode='r')) for s in ['train','validation','test']}
    for split,(x,y) in arrays.items():
        assert x.shape==(info['sizes'][split],info['features']) and len(y)==len(x)
    cache={}
    #This changes storage only: cache entire train/validation splits when small.
    for split in ['train','validation']:
        x,y=arrays[split]
        if x.nbytes+y.nbytes <= 128*1024*1024:
            cache[split]=(pt.tensor(np.asarray(x),device=device),pt.tensor(np.asarray(y),device=device))
    def batch(split,indices):
        if split in cache:
            idx=indices if isinstance(indices,slice) else pt.as_tensor(indices,device=device)
            return cache[split][0][idx],cache[split][1][idx]
        x,y=arrays[split]
        return pt.from_numpy(np.array(x[indices],copy=True)).to(device),pt.from_numpy(np.array(y[indices],copy=True)).to(device)
    pt.manual_seed(seed)
    config=module.SUANNconfig(128,definitions.numberOfLayers,0,definitions.hiddenLayerSize,None,
        info['features'],info['class_count'],1,info['features'],info['class_count'],info['sizes']['train'],None)
    model=module.SUANNmodel(config).to(device)
    assert not list(model.buffers()),'Batched evaluator only supports stateless tabular MLPs'
    initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=pt.optim.Adam(model.parameters(),lr=definitions.learningRate) if method=='backprop_adam' else None
    initial_lr=definitions.learningRate if optimizer is not None else PROTOCOL['population_learning_rate']
    chunk=None;fast=None
    if optimizer is None:
        n=int(method.split('_')[1]);algorithm.populationPertubationPopulationSize=n
        algorithm.populationPertubationSigma=PROTOCOL['sigma'];algorithm.populationPertubationLearningRate=initial_lr
        chunk=min(n,max(8,min(24_000_000//parameters,17_000_000//(128*max(definitions.hiddenLayerSize,info['class_count'])))))
        fast=CachedGraphedPopulationStep(model,algorithm,seed+300000,chunk)
    repeats=definitions.datasetRepeatSize if definitions.datasetRepeat else 1
    train_count=info['sizes']['train'];epoch_rows=train_count*repeats
    controller=convergence.PopulationPertubationTrainingConvergence(initial_lr,(epoch_rows+127)//128)
    batch_generator=pt.Generator().manual_seed(seed+100000)
    order=pt.empty(0,dtype=pt.long);cursor=0;passes=0;examples=0
    schedule_chain='00'*32
    def next_batch():
        nonlocal order,cursor,passes,examples,schedule_chain
        pieces=[];needed=128
        while needed:
            if cursor==len(order):
                order=pt.randperm(epoch_rows,generator=batch_generator)%train_count;cursor=0;passes+=1
            take=min(needed,len(order)-cursor);pieces.append(order[cursor:cursor+take]);cursor+=take;needed-=take
        indices=pt.cat(pieces);examples+=128
        schedule_chain=hashlib.sha256(bytes.fromhex(schedule_chain)+indices.numpy().tobytes()).hexdigest()
        return indices.numpy()
    pt.manual_seed(seed+200000)
    @pt.no_grad()
    def evaluate(split):
        model.eval();model.accuracyFunction.reset()
        total=0.;correct=0;n=info['sizes'][split]
        counts=pt.zeros(info['class_count'],dtype=pt.int64,device=device);hits=pt.zeros_like(counts)
        for start in range(0,n,PROTOCOL['evaluation_batch_size']):
            x,y=batch(split,slice(start,min(n,start+PROTOCOL['evaluation_batch_size'])))
            loss,_=algorithm.trainOrTestModel(model,False,x,y,None,None)
            match=model.Ztrace[-1].argmax(1)==y
            total+=loss.item()*len(y);correct+=match.sum().item()
            counts+=pt.bincount(y,minlength=len(counts));hits+=pt.bincount(y[match],minlength=len(counts))
        return {'loss':total/n,'accuracy':correct/n,'balanced_accuracy':(hits[counts>0].double()/counts[counts>0]).mean().item(),'rows':n}
    checkpoint=RUNS/f'{tag}.pt';WORK.mkdir(parents=True,exist_ok=True);working=WORK/f'{tag}.pt'
    candidates=[p for p in [checkpoint,working] if p.exists()]
    step=0;curves=[];training_seconds=0.;elapsed_saved=0.;stopping_train=None;stopping_validation=None
    if candidates:
        state=pt.load(max(candidates,key=lambda p:p.stat().st_mtime_ns),map_location='cpu',weights_only=False)
        assert state['protocol']==PROTOCOL and state['data_sha256']==digest
        assert state['initial_parameters_sha256']==initial_hash
        model.load_state_dict(state['model']);controller.load_state_dict(state['controller'])
        if optimizer is not None:optimizer.load_state_dict(state['optimizer'])
        if fast is not None:
            assert state['chunk']==chunk
            fast.generator.set_state(state['noise_rng']);algorithm.populationPertubationLearningRate=controller.learningRate
        order=state['order'];cursor=state['cursor'];passes=state['passes'];examples=state['examples'];schedule_chain=state['schedule_chain']
        batch_generator.set_state(state['batch_rng']);pt.set_rng_state(state['cpu_rng']);pt.cuda.set_rng_state_all(state['cuda_rng'])
        step=state['step'];curves=state['curves'];training_seconds=state['training_seconds'];elapsed_saved=state['elapsed_seconds'];stopping_train=state['stopping_train'];stopping_validation=state['stopping_validation']
    start=time.perf_counter();last_backup=0.
    def save(force=False):
        nonlocal last_backup
        state={'protocol':PROTOCOL,'data_sha256':digest,'initial_parameters_sha256':initial_hash,
            'model':convergence._cpuCopy(model.state_dict()),'optimizer':convergence._cpuCopy(optimizer.state_dict()) if optimizer is not None else None,
            'controller':controller.state_dict(),'noise_rng':fast.generator.get_state() if fast else None,'chunk':chunk,
            'order':order,'cursor':cursor,'passes':passes,'examples':examples,'schedule_chain':schedule_chain,
            'batch_rng':batch_generator.get_state(),'cpu_rng':pt.get_rng_state(),'cuda_rng':pt.cuda.get_rng_state_all(),
            'step':step,'curves':curves,'training_seconds':training_seconds,'elapsed_seconds':elapsed_saved+time.perf_counter()-start,'stopping_train':stopping_train,'stopping_validation':stopping_validation}
        tmp=working.with_suffix('.tmp');pt.save(state,tmp);tmp.replace(working)
        if force or time.perf_counter()-last_backup>=300:
            tmp=checkpoint.with_suffix('.tmp');shutil.copyfile(working,tmp);tmp.replace(checkpoint);last_backup=time.perf_counter()
    def progress(train=None,validation=None,status='training'):
        record={'dataset':dataset,'method':method,'seed':seed,'status':status,'step':step,'lr':controller.learningRate,
            'trainSetLossOptimisation':controller.trainSetLossOptimisation,'selection':controller.selection,
            'stage':controller.reductions,'train':train,'validation':validation,'best_step':controller.best['step'] if controller.best else None,
            'best_train':controller.best['train'] if controller.best else None,'training_rows':train_count,
            'best_validation':controller.best['validation'] if controller.best else None,
            'examples_processed':examples,'effective_training_passes':examples/train_count,'minimum_updates':controller.policy['minimum_updates'],
            'numerical_recoveries':controller.numericalRecoveries,'stop_reason':controller.stopReason,'time':time.time()}
        atomic_json(output.with_suffix('.progress.json'),record)
    progress(status='evaluating_initial_full_training_set' if step==0 else 'resumed')
    if controller.best is None:
        metrics=evaluate('train')
        if controller.trainSetLossOptimisation:
            #Preserve the original order: select initial train checkpoint before diagnostic validation.
            controller.observe(0,metrics,None,model,optimizer)
        validation=evaluate('validation')
        if controller.trainSetLossOptimisation:
            controller.best['validation']=validation
        else:
            controller.observe(0,metrics,validation,model,optimizer)
        curves.append({'step':0,'lr':controller.learningRate,'train':metrics,'validation':validation,'schedule_chain':schedule_chain})
        save(force=True);progress(metrics,validation)
    while controller.stopReason is None:
        idx=next_batch();x,y=batch('train',idx);model.train();pt.cuda.synchronize();tick=time.perf_counter();step+=1
        error=None
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss,_=algorithm.trainOrTestModel(model,True,x,y,optimizer,None)
                loss.backward();optimizer.step()
            else:loss,_=fast(x,y)
            if not pt.isfinite(loss).item():raise FloatingPointError('Non-finite reported training loss')
        except FloatingPointError as caught:error=caught
        pt.cuda.synchronize();training_seconds+=time.perf_counter()-tick
        if error is not None:
            controller.recoverNonfinite(step,model,optimizer,str(error))
            if optimizer is None:algorithm.populationPertubationLearningRate=controller.learningRate
            save(force=True);progress(status='numerical_recovery');continue
        if not controller.shouldEvaluate(step):continue
        progress(status='evaluating_full_training_set')
        try:
            metrics=evaluate('train')
            point={'step':step,'lr':controller.learningRate,'stage':controller.reductions,'train':metrics,
                'validation':evaluate('validation'),'training_seconds':training_seconds,'elapsed_seconds':elapsed_saved+time.perf_counter()-start,
                'schedule_chain':schedule_chain}
            controller.observe(step,metrics,point['validation'],model,optimizer)
        except FloatingPointError as error:
            controller.recoverNonfinite(step,model,optimizer,str(error))
            if optimizer is None:algorithm.populationPertubationLearningRate=controller.learningRate
            save(force=True);progress(status='numerical_recovery');continue
        curves.append(point);stopping_train=metrics;stopping_validation=point['validation']
        if optimizer is None:algorithm.populationPertubationLearningRate=controller.learningRate
        progress(metrics,point['validation']);save(force=controller.stopReason is not None)
        if step%1000==0:print(tag,'step',step,'train',metrics,'lr',controller.learningRate,flush=True)
    controller.restoreBest(model,optimizer)
    #The first test-set evaluation occurs only after stopping and checkpoint selection.
    progress(status='evaluating_selected_checkpoint')
    result={'dataset':dataset,'method':method,'seed':seed,'status':'complete','protocol':PROTOCOL,'steps':step,
        'selected_step':controller.best['step'],'selection':controller.selection,'stop_reason':controller.stopReason,
        'train':evaluate('train'),'validation':evaluate('validation'),'test':evaluate('test'),'stopping_train':stopping_train,'stopping_validation':stopping_validation,
        'data_sha256':digest,'initial_parameters_sha256':initial_hash,'schedule_chain':schedule_chain,'training_rows':train_count,
        'parameters':parameters,'architecture':[info['features']]+[definitions.hiddenLayerSize]*(definitions.numberOfLayers-1)+[info['class_count']],
        'initial_learning_rate':initial_lr,'final_learning_rate':controller.learningRate,'sigma':None if optimizer is not None else PROTOCOL['sigma'],
        'convergence_policy':controller.policy,'learning_rate_events':controller.events,'curves':curves,'training_seconds':training_seconds,
        'elapsed_seconds':elapsed_saved+time.perf_counter()-start,'examples_processed':examples,'population_chunk_size':chunk,
        'backend':'Adam' if fast is None else type(fast).__name__,'torch_version':pt.__version__,'device':pt.cuda.get_device_name(),
        'global_optimum_proven':False,'full_source_used':True}
    selected=RUNS/f'{tag}.best.pt';tmp=selected.with_suffix('.tmp');pt.save(controller.best,tmp);tmp.replace(selected)
    atomic_json(output,result);progress(result['train'],result['validation'],'complete')
    print('COMPLETE',tag,step,'train',result['train'],'validation',result['validation'],'test',result['test'],flush=True)
    return result

def main():
    require_workspace()
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--datasets',nargs='+',default=PROTOCOL['datasets']);parser.add_argument('--methods',nargs='+',default=PROTOCOL['methods'])
    parser.add_argument('--seeds',nargs='+',type=int,default=PROTOCOL['seeds']);args=parser.parse_args()
    missing=[d for d in args.datasets if not (HERE/'data'/d/'manifest.json').exists()]
    if missing:raise RuntimeError(f'Full datasets not prepared: {missing}; no capped fallback is permitted')
    jobs=[(d,m,s) for m in args.methods for d in args.datasets for s in args.seeds if not (RUNS/f'{d}_{m}_seed{s}.json').exists()]
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        futures={pool.submit(run_one,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception:
                error={'job':futures[future],'traceback':traceback.format_exc()};failures.append(error)
                atomic_json(HERE/'logs'/('failure_'+'_'.join(map(str,futures[future]))+'.json'),error)
                print('FAILURE',error,flush=True)
    if failures:raise RuntimeError(f'{len(failures)} failed runs; inspect logs')

if __name__=='__main__':main()
