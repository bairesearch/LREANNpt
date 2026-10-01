import os,json,time,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['SUANN_REPOSITORY']=str(HERE/'source')
import torch as pt
from model_setup import load_modules
from fast_population import PopulationStep
pt.set_num_threads(1);pt.set_num_interop_threads(1)
_,algorithm,_=load_modules('population_64','titanic')
module=algorithm.LREANNpt_SUANNmodel;module.device=pt.device('cuda')
pt.manual_seed(314)
model=module.SUANNmodel(module.SUANNconfig(128,4,0,32,None,7,2,1,7,2,1000,None)).cuda()
x=pt.randn(128,7,device='cuda');y=pt.randint(2,(128,),device='cuda')
original=copy.deepcopy(model.state_dict());helper=algorithm.generate_population_pertubation_noise
checks=[]
for count in (64,256,1024,4096):
    algorithm.populationPertubationPopulationSize=count
    fast=PopulationStep(model,algorithm,99,64)
    noise=fast.noises(list(model.parameters()),count)
    pt.manual_seed(505)
    seeds=pt.randint(0,2**63-1,(count,),device='cpu').tolist()
    indices={seed:i for i,seed in enumerate(seeds)}
    def injected(parameters,seed):
        return (e[indices[seed]] for e in noise)
    algorithm.generate_population_pertubation_noise=injected
    model.load_state_dict(original);pt.manual_seed(505)
    algorithm.trainOrTestModel(model,True,x,y,None,None)
    reference=[p.clone() for p in model.parameters()]
    model.load_state_dict(original)
    fast(x,y,provided_noise=noise)
    error=max((p-q).abs().max().item() for p,q in zip(model.parameters(),reference))
    for p,q in zip(model.parameters(),reference):pt.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
    checks.append({'population':count,'max_parameter_error':error})
algorithm.generate_population_pertubation_noise=helper
# Check RNG/noise/model restart identity across several consecutive updates.
algorithm.populationPertubationPopulationSize=256
fast=PopulationStep(model,algorithm,100,64);model.load_state_dict(original)
saved_rng=fast.generator.get_state();fast(x,y);fast(x,y)
expected=copy.deepcopy(model.state_dict());end_rng=fast.generator.get_state()
model.load_state_dict(original);fast.generator.set_state(saved_rng);fast(x,y);fast(x,y)
assert pt.equal(end_rng,fast.generator.get_state())
assert all(pt.equal(v,expected[k]) for k,v in model.state_dict().items())
# Informational throughput after warmup, same real forward.
timings=[]
for count in (64,256,1024,4096):
    algorithm.populationPertubationPopulationSize=count
    model.load_state_dict(original);fast=PopulationStep(model,algorithm,101,128)
    fast(x,y);pt.cuda.synchronize();start=time.perf_counter()
    for _ in range(5):fast(x,y)
    pt.cuda.synchronize();timings.append({'population':count,'seconds_per_update':(time.perf_counter()-start)/5})
result={'same_noise_production_comparisons':checks,'rng_resume_bitwise_equal':True,'fixture_architecture':[7,32,32,32,2],'timings':timings}
(HERE/'verification/backend.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
