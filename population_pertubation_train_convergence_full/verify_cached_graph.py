import os,json,copy,time,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;os.environ['SUANN_REPOSITORY']=str(HERE/'source')
import torch as pt
from model_setup import load_modules
from fast_population import PopulationStep
from cached_graphed_population import CachedGraphedPopulationStep as GraphedPopulationStep
pt.set_num_threads(1);pt.set_num_interop_threads(1)
parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='iris');args=parser.parse_args()
definitions,algorithm,_=load_modules('population_64',args.dataset);module=algorithm.LREANNpt_SUANNmodel;module.device=pt.device('cuda')
features,classes=(4,3) if args.dataset=='iris' else (54,7)
pt.manual_seed(222);model=module.SUANNmodel(module.SUANNconfig(128,definitions.numberOfLayers,0,definitions.hiddenLayerSize,None,features,classes,1,features,classes,10000,None)).cuda()
x=pt.randn(128,features,device='cuda');y=pt.randint(classes,(128,),device='cuda');initial=copy.deepcopy(model.state_dict())
count_params=sum(p.numel() for p in model.parameters());checks=[]
for count in ([64,256,1024,4096] if args.dataset=='iris' else [64,256]):
 algorithm.populationPertubationPopulationSize=count
 for chunk in sorted({min(count,128),min(count,max(8,min(24_000_000//count_params,17_000_000//(128*max(definitions.hiddenLayerSize,classes)))))}):
  model.load_state_dict(initial);algorithm.populationPertubationLearningRate=.01
  eager=PopulationStep(model,algorithm,979,chunk);reference=[]
  for i in range(6):
   if i==3:algorithm.populationPertubationLearningRate=.002
   loss,_=eager(x+i*.01,y)
   reference.append(([p.clone() for p in model.parameters()],eager.generator.get_state(),loss.item()))
  model.load_state_dict(initial);algorithm.populationPertubationLearningRate=.01
  graph=GraphedPopulationStep(model,algorithm,979,chunk);errors=[]
  for i,(parameters,rng,loss) in enumerate(reference):
   if i==3:algorithm.populationPertubationLearningRate=.002
   observed,_=graph(x+i*.01,y)
   error=max((p-q).abs().max().item() for p,q in zip(model.parameters(),parameters));errors.append(error)
   for p,q in zip(model.parameters(),parameters):pt.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
   assert pt.equal(graph.generator.get_state(),rng),'RNG stream mismatch'
   assert observed.item()==loss and error==0, 'Expected bitwise equality'
   if i==2:
    model.eval()
    with pt.no_grad():model(False,x.repeat(3,1),y.repeat(3),None,None)
    model.train()
  # Restore both parameter and graph generator state and repeat two updates exactly.
  saved=copy.deepcopy(model.state_dict());rng=graph.generator.get_state();graph(x,y);graph(x,y);expected=copy.deepcopy(model.state_dict());endrng=graph.generator.get_state()
  model.load_state_dict(saved);graph.generator.set_state(rng);graph(x,y);graph(x,y)
  assert all(pt.equal(v,expected[k]) for k,v in model.state_dict().items()) and pt.equal(endrng,graph.generator.get_state())
  model.load_state_dict(saved);fresh=GraphedPopulationStep(model,algorithm,1,chunk);fresh.generator.set_state(rng);fresh(x,y);fresh(x,y)
  assert all(pt.equal(v,expected[k]) for k,v in model.state_dict().items()) and pt.equal(endrng,fresh.generator.get_state())
  del fresh
  timings={}
  for name,backend in [('eager',eager),('graph',graph)]:
   backend(x,y);pt.cuda.synchronize();start=time.perf_counter()
   for _ in range(30):backend(x,y)
   pt.cuda.synchronize();timings[name]=(time.perf_counter()-start)/30
  checks.append({'population':count,'chunk':chunk,'max_error_over_six_updates':max(errors),'rng_exact':True,'resume_exact':True,'fresh_capture_resume_exact':True,'evaluation_between_updates_verified':True,'timings':timings})
  print(checks[-1],flush=True)
  del graph,eager;pt.cuda.empty_cache()
(HERE/'verification'/f'cached_cuda_graph_{args.dataset}.json').write_text(json.dumps(checks,indent=2))
