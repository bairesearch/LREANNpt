#!/usr/bin/env python3
"""Live full-data results; never report incomplete runs as final test measurements."""
from pathlib import Path
import argparse, csv, datetime, fcntl, json, math, statistics
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
COLORS={'backprop_adam':'#152238','population_64':'#95C13D','population_256':'#168F82','population_1024':'#2B6B80','population_4096':'#493474'}

def atomic_text(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(value);tmp.replace(path)

def records():
    completed=[];active=[]
    for path in (HERE/'runs').glob('*.json'):
        if path.name.endswith('.progress.json'):
            tag=path.name.removesuffix('.progress.json')
            if not (HERE/'runs'/f'{tag}.json').exists():active.append(json.loads(path.read_text()))
        else:
            record=json.loads(path.read_text())
            if record['protocol']!=P:
                raise RuntimeError(f'{path.name}: result belongs to an earlier protocol; start a fresh reproduction before reporting the bank-full experiment')
            completed.append(record)
    failures=[json.loads(p.read_text()) for p in (HERE/'logs').glob('failure_*.json')]
    return completed,active,failures

def average(values):return statistics.mean(values)
def sd(values):return statistics.stdev(values) if len(values)>1 else 0.

def main():
    report_lock=(HERE/'logs/report.lock').open('a')
    fcntl.flock(report_lock,fcntl.LOCK_EX)
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true');args=parser.parse_args()
    rows,active,failures=records();groups={(d,m):[r for r in rows if r['dataset']==d and r['method']==m] for d in P['datasets'] for m in P['methods']}
    manifests={d:json.loads((HERE/'data'/d/'manifest.json').read_text()) for d in P['datasets'] if (HERE/'data'/d/'manifest.json').exists()}
    held=json.loads((HERE/'HOLDS.json').read_text()) if (HERE/'HOLDS.json').exists() else {}
    now=datetime.datetime.now().astimezone().isoformat()
    progress={'time':now,'completed':len(rows),'expected':195,'active':active,'failures':failures,'held_datasets':held}
    atomic_text(HERE/'PROGRESS.json',json.dumps(progress,indent=2)+'\n')
    lines=['SUANN population perturbation: full-data training convergence',f'Updated: {now}',f'Completed runs: {len(rows)}/195; unresolved run failures: {len(failures)}','',
        'Scope and protocol',P['scope'],'No training, validation, test, or source-prefix row caps. All runs start from fresh models.',
        '13 tabular datasets; Adam backprop and population sizes 64, 256, 1024, 4096; paired seeds 11, 22, 33.',
        'The cancelled capped experiment is not resumed and its results are not pooled with these results.',
        'Dataset partition rules: '+P['split_rule'],
        'HIGGS uses all 11 million rows with a random 60/20/20 partition, not the original paper final-500,000 test partition.',
        'All train/validation/test source indices and input-source/prepared-file hashes are saved under data/.',
        'Categories and min/max normalization are fitted on the full training split. The streamed float32 transformation is checked against patched production ANNpt_data.',
        'Training convergence',
        'Full training cross-entropy is evaluated every 100 updates. Significant decrease: greater than max(0.0001, 0.001 * last significant best loss).',
        'Minimum training: '+P['minimum_updates_rule'],
        'After 2,000 updates without significant improvement, restore the minimum-training-CE model and its optimizer and multiply learning rate by 0.2.',
        P['stop'],
        'The selected model has the lowest observed full-training CE; it may differ from the stopping model in accuracy.',
        'Validation is logged only. Test evaluation occurs only after the stopping decision and restoration of the selected checkpoint.',
        'Numerical recovery restores the finite best model and reduces learning rate by 0.2, up to five recoveries. Recoveries count as learning-rate reductions; numerical failure itself is never convergence.',
        'This is a practical training-loss convergence experiment; no mathematical global optimum is claimed.',
        'Model/settings: repository architecture and Adam learning rate per dataset; population learning rate 0.01, sigma 0.01, batch size 128.',
        'Every trainable parameter is perturbed jointly. Candidate reward is negative minibatch CE. Update is alpha/(N*sigma) * sum((reward_i - mean_reward) * epsilon_i).',
        'The benchmark batches the actual SUANN forward with iid Gaussian directions, verified with matched noise against sequential production. The random-number stream differs from sequential production.',
        'All methods share initial model seed and a separate identical shuffled minibatch stream per dataset/seed. Remainders carry into the next pass: no rows are dropped.',
        'Production opt-in: populationPertubationOptimiseTrainingIterations=True with useStochasticUpdates=True, useIndividualPertubation=False, usePopulationPertubation=True.',
        'The benchmark also enables the same stopping controller for Adam, with useStochasticUpdates=False.',
        'Production uses its full training DataLoader including a partial final batch; the benchmark carries remainders to keep batches of 128 for paired GPU evaluation.',
        'Timing: three concurrent GPU workers; durations include different workloads and are not isolated optimizer speed comparisons.',
        'Blog Feedback retains raw integer comment-count class labels, matching the repository; accuracy is not a regression metric.',
        P['banking_marketing_policy'],
        '','Dataset sizes (no row subsampling)', '| Dataset | Source rows | Training | Validation | Test | Features |', '|---|---:|---:|---:|---:|---:|']
    for d in P['datasets']:
        if d in manifests:
            v=manifests[d];s=v['sizes'];lines.append(f"| {d} | {v['source_rows_loaded']:,} | {s['train']:,} | {s['validation']:,} | {s['test']:,} | {v['features']} |")
    if held:
        lines+=['','Dataset holds']+[f'{d}: {reason}' for d,reason in held.items()]
    integrity=HERE/'verification/source_split_overlap.json'
    if integrity.exists():lines+=['','Source split integrity',integrity.read_text()]
    lines+=['','Test-set accuracy (%) after training, mean ± sample SD over three seeds. A cell remains pending until all three seeds finish.',
            '| Dataset | Backprop Adam | Population 64 | 256 | 1024 | 4096 |','|---|---:|---:|---:|---:|---:|']
    summary=[]
    for d in P['datasets']:
        cells=[]
        for m in P['methods']:
            group=groups[d,m]
            if len(group)==3:
                metrics={f'{split}_{metric}_{stat}':fn([r[split][metric] for r in group]) for split in ['train','validation','test'] for metric in ['accuracy','loss','balanced_accuracy'] for stat,fn in [('mean',average),('sd',sd)]}
                summary.append({'dataset':d,'method':m,'seeds':3,**metrics,'minimum_steps':min(r['steps'] for r in group),'maximum_steps':max(r['steps'] for r in group)})
                cells.append(f"{metrics['test_accuracy_mean']*100:.2f} ± {metrics['test_accuracy_sd']*100:.2f}")
            else:cells.append(f'pending ({len(group)}/3)')
        lines.append('| '+d+' | '+' | '.join(cells)+' |')
    lines+=['','Completed groups: training fit and stopping iterations','| Dataset | Method | Train accuracy (%) | Train CE | Test balanced accuracy (%) | Stopping updates |','|---|---|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['dataset']} | {s['method']} | {100*s['train_accuracy_mean']:.2f} | {s['train_loss_mean']:.6f} | {100*s['test_balanced_accuracy_mean']:.2f} | {s['minimum_steps']}–{s['maximum_steps']} |")
    lines+=['','Active runs']
    for r in active:lines.append(f"{r['dataset']} {r['method']} seed {r['seed']}: step {r['step']}; {r['status']}; train={r.get('train')}; best_train={r.get('best_train')}")
    lines+=['','Numerical recovery and learning-rate events']
    for r in rows:
        for e in r['learning_rate_events']:lines.append(f"{r['dataset']} {r['method']} seed {r['seed']}: {json.dumps(e)}")
    lines+=['','Unresolved failures']+[json.dumps(f) for f in failures]
    lines+=['','Artifacts','protocol.json: frozen experiment protocol and production source hashes.','data/<dataset>/manifest.json and *_source_indices.npy: uncapped split/source/preprocessing provenance.',
        'runs/*.pt: resumable state; runs/*.best.pt: minimum training-CE checkpoint; runs/*.json: completed results.',
        'verification/: numerical, data coverage, and independent final metric checks.',
        'training_loss_grid.png/svg, validation_loss_grid.png/svg, test_accuracy_by_population.png/svg, figures/: scientific plots.',
        'CANCEL: create this file to stop this experiment and its worker process group. No cancelled experiment is automatically restarted.','']
    atomic_text(HERE/'REPORT.txt','\n'.join(lines))
    atomic_text(HERE/'summary.json',json.dumps(summary,indent=2)+'\n')
    if summary:
        with (HERE/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    if rows:
        fields=['dataset','method','seed','steps','selected_step','stop_reason','training_rows','training_seconds','elapsed_seconds']
        with (HERE/'per_seed.csv').open('w',newline='') as f:
            columns=fields+[f'{s}_{v}' for s in ['train','validation','test'] for v in ['accuracy','loss','balanced_accuracy']]
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            for r in rows:writer.writerow({**{k:r[k] for k in fields},**{f'{s}_{v}':r[s][v] for s in ['train','validation','test'] for v in ['accuracy','loss','balanced_accuracy']}})
    if args.plots:plot(rows,groups,summary)
    print(json.dumps({'completed':len(rows),'expected':195,'active':len(active),'failures':len(failures),'time':now}),flush=True)

def plot(rows,groups,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    def save(fig,path):
        for ext in ['png','svg']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight')
        plt.close(fig)
    def loss_axis(ax,dataset,split,logx=False):
        for method in P['methods']:
            group=groups[dataset,method];color=COLORS[method]
            curves=[{p['step']:p[split]['loss'] for p in r['curves'] if p['step']>0 and math.isfinite(p[split]['loss'])} for r in group]
            for r,c in zip(group,curves):
                ax.plot(list(c),list(c.values()),color=color,alpha=.25,lw=.7)
                if split=='train' and r['selected_step'] in c:ax.plot(r['selected_step'],c[r['selected_step']],'o',color=color,ms=3)
            if curves:
                common=sorted(set.intersection(*(set(c) for c in curves)))
                if common:ax.plot(common,[np.mean([c[s] for c in curves]) for s in common],color=color,lw=1.7,ls='--' if method=='backprop_adam' else '-',label=method.replace('population_','N='))
        ax.set_title(dataset);ax.set_xlabel('Training updates');ax.set_ylabel(f'{split.capitalize()} cross-entropy');ax.grid(alpha=.2)
        if logx:ax.set_xscale('log')
        if ax.lines:ax.legend(fontsize=7)
    for split in ['train','validation']:
        fig,axes=plt.subplots(4,4,figsize=(17,13),constrained_layout=True)
        for ax,d in zip(axes.flat,P['datasets']):loss_axis(ax,d,split,True)
        for ax in list(axes.flat)[13:]:ax.set_visible(False)
        fig.suptitle(f'SUANN full-data {split} loss — {len(rows)}/195 runs complete\nThin lines: seeds; bold: mean over shared recorded updates',fontsize=15)
        save(fig,HERE/('training_loss_grid' if split=='train' else 'validation_loss_grid'))
        if len(rows)==195:
            for d in P['datasets']:
                fig,ax=plt.subplots(figsize=(8,5));loss_axis(ax,d,split);save(fig,HERE/'figures'/f'{d}_{split}_loss')
    fig,axes=plt.subplots(4,4,figsize=(17,13),constrained_layout=True)
    for ax,d in zip(axes.flat,P['datasets']):
        data={s['method']:s for s in summary if s['dataset']==d}
        pops=[n for n in [64,256,1024,4096] if f'population_{n}' in data]
        if pops:ax.errorbar(pops,[100*data[f'population_{n}']['test_accuracy_mean'] for n in pops],yerr=[100*data[f'population_{n}']['test_accuracy_sd'] for n in pops],fmt='o-',color='#168F82',capsize=3,label='Population mean ± SD')
        if 'backprop_adam' in data:
            b=data['backprop_adam'];mean=100*b['test_accuracy_mean'];sdv=100*b['test_accuracy_sd']
            ax.axhline(mean,color='#152238',ls='--',label='Adam mean ± SD');ax.axhspan(mean-sdv,mean+sdv,color='#152238',alpha=.1)
        ax.set_xscale('log',base=4);ax.set_xticks([64,256,1024,4096],['64','256','1024','4096']);ax.set_xlim(48,5500)
        ax.set_title(d);ax.set_xlabel('Perturbation population size');ax.set_ylabel('Test accuracy (%)');ax.grid(alpha=.2)
        if data:ax.legend(fontsize=7)
    for ax in list(axes.flat)[13:]:ax.set_visible(False)
    fig.suptitle(f'Full-data test accuracy after training convergence — {len(rows)}/195 complete\nThree paired seeds; checkpoint selected using training loss only',fontsize=15)
    save(fig,HERE/'test_accuracy_by_population')

if __name__=='__main__':main()
