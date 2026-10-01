#!/usr/bin/env python3
"""Run the authorized uncapped queue, monitor it, then verify and plot results.

Creating HERE/CANCEL or sending SIGTERM stops the entire worker process group.
Held datasets are not submitted until their explicit hold is removed.
"""
from pathlib import Path
import concurrent.futures, datetime, fcntl, json, multiprocessing as mp, os, signal, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
from reproduction_support import require_workspace

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def status(stage,**values):atomic_json(HERE/'MANAGER.json',{'stage':stage,'time':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid(),**values})

def cancel(signum=None,frame=None):
    status('cancelled_by_user')
    #The detached manager owns its session/process group; only its workers belong here.
    os.killpg(os.getpgrp(),signal.SIGKILL)

def main():
    require_workspace()
    lock=(HERE/'logs/manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.getpgrp()==os.getpid(),'Launch with start_new_session=True for isolated cancellation'
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    if (HERE/'CANCEL').exists():cancel()
    for name in ['backend.json','cached_cuda_graph_iris.json','preprocessing_equivalence.json','full_data_audit.json']:
        assert (HERE/'verification'/name).exists(),f'Missing verification gate: {name}'
    from train_full import run_one
    submitted=set();failures=[];last_report=0.;last_plot=-1
    with concurrent.futures.ProcessPoolExecutor(max_workers=P['concurrent_workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        futures={}
        while True:
            if (HERE/'CANCEL').exists():cancel()
            holds=json.loads((HERE/'HOLDS.json').read_text()) if (HERE/'HOLDS.json').exists() else {}
            for method in P['methods']:
                for dataset in P['datasets']:
                    if dataset in holds:continue
                    assert (HERE/'data'/dataset/'manifest.json').exists()
                    for seed in P['seeds']:
                        job=(dataset,method,seed)
                        if job in submitted:continue
                        submitted.add(job)
                        output=HERE/'runs'/f'{dataset}_{method}_seed{seed}.json'
                        if output.exists():
                            saved=json.loads(output.read_text());assert saved['status']=='complete' and saved['protocol']==P
                            if not output.with_suffix('.pt').exists() or not output.with_suffix('.best.pt').exists():
                                raise RuntimeError(f'{output.name}: completed result is missing checkpoints; '
                                                   'start a fresh reproduction instead of resuming a cleaned archive')
                        else:futures[pool.submit(run_one,job)]=job
            for future in list(futures):
                if not future.done():continue
                job=futures.pop(future)
                try:future.result()
                except Exception:
                    error={'job':job,'traceback':traceback.format_exc()};failures.append(error)
                    atomic_json(HERE/'logs'/('failure_'+'_'.join(map(str,job))+'.json'),error)
                    print('FAILURE',json.dumps(error),flush=True)
            if time.time()-last_report>=60:
                subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
                status('training' if futures else 'waiting_for_dataset_decision',submitted=len(submitted),pending=len(futures),failures=len(failures),held_datasets=holds)
                last_report=time.time()
            if not futures and not holds:break
            time.sleep(5)
    if failures:
        status('failed_runs_require_review',failures=failures)
        raise RuntimeError(f'{len(failures)} training runs failed; inspect logs')
    status('independent_final_verification')
    subprocess.run([sys.executable,str(HERE/'audit_data.py')],check=True)
    with (HERE/'logs/final_metrics.log').open('a') as f:
        subprocess.run([sys.executable,str(HERE/'verify_results.py'),'--require-complete'],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(HERE/'report.py'),'--plots'],check=True)
    from PIL import Image
    import xml.etree.ElementTree as ET
    pngs=list(HERE.glob('*.png'))+list((HERE/'figures').glob('*.png'));svgs=list(HERE.glob('*.svg'))+list((HERE/'figures').glob('*.svg'))
    assert len(pngs)==len(svgs)==29
    for path in pngs:
        with Image.open(path) as image:image.verify()
    for path in svgs:ET.parse(path)
    status('automated_checks_complete',completed=195,visual_review_pending=True,verified_pngs=len(pngs),verified_svgs=len(svgs))

if __name__=='__main__':
    try:main()
    except Exception:
        status('manager_error',traceback=traceback.format_exc());raise
