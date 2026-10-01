"""Benchmark-only GPU batching of the population estimator.

Uses SUANN's real forward function and its exact seed/noise replay helper.
Only tabular MLPs without mutable buffers or dropout are supported. This is
an evaluation backend, not a new optimizer; production files are unchanged.
"""
import torch as pt


@pt.no_grad()
def population_step(model, algorithm, x, y, chunk_size=128):
    if not algorithm.usePopulationPertubation:
        raise ValueError('Population perturbation must be enabled')
    if list(model.buffers()) or any(isinstance(m, (pt.nn.Dropout, pt.nn.modules.batchnorm._BatchNorm)) for m in model.modules()):
        raise ValueError('The benchmark batching backend requires a stateless tabular MLP')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and p.numel()]
    names, parameters = zip(*named)
    originals = [p.detach().clone() for p in parameters]
    count = algorithm.populationPertubationPopulationSize
    sigma = algorithm.populationPertubationSigma
    rate = algorithm.populationPertubationLearningRate
    seeds = pt.randint(0, 2**63-1, (count,), device='cpu').tolist()
    rewards = []
    had_flag = hasattr(model, '_populationPertubationEvaluating')
    flag = getattr(model, '_populationPertubationEvaluating', False)

    def loss_for_params(params):
        return pt.func.functional_call(model, params, (True, x, y, None, None))[0]

    def chunk_noise(chunk):
        members = [list(algorithm.generate_population_pertubation_noise(parameters, seed)) for seed in chunk]
        return [pt.stack([member[index] for member in members]) for index in range(len(parameters))]

    try:
        model._populationPertubationEvaluating = True
        for start in range(0, count, chunk_size):
            noise = chunk_noise(seeds[start:start+chunk_size])
            candidates = {name: original.unsqueeze(0) + sigma*epsilon for name, original, epsilon in zip(names, originals, noise)}
            rewards.append(-pt.vmap(loss_for_params)(candidates))
        reward = pt.cat(rewards)
        if not pt.isfinite(reward).all().item():
            raise FloatingPointError('Nonfinite candidate loss')
        #Float64 centring matches the scalar implementation's Python accumulation.
        weights = ((reward.double() - reward.double().mean()) * (rate / (count*sigma))).float()
        updates = [pt.zeros_like(p) for p in parameters]
        for start in range(0, count, chunk_size):
            noise = chunk_noise(seeds[start:start+chunk_size])
            weight = weights[start:start+chunk_size]
            for update, epsilon in zip(updates, noise):
                update.add_(pt.einsum('n,n...->...', weight, epsilon))
        candidates = [original+update for original, update in zip(originals, updates)]
        if not all(pt.isfinite(candidate).all().item() for candidate in candidates):
            raise FloatingPointError('Nonfinite update')
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
    finally:
        if had_flag:
            model._populationPertubationEvaluating = flag
        else:
            del model._populationPertubationEvaluating
    return model(True, x, y, None, None)


if __name__ == '__main__':
    import copy
    import time
    from model_setup import load_modules
    pt.set_num_threads(1)
    _, algorithm, _ = load_modules('population_64', 'titanic')
    module = algorithm.LREANNpt_SUANNmodel
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    module.device = device
    pt.manual_seed(314)
    model = module.SUANNmodel(module.SUANNconfig(128,4,0,128,None,7,2,1,7,2,1000,None)).to(device)
    x,y=pt.randn(128,7,device=device),pt.randint(0,2,(128,),device=device)
    original=copy.deepcopy(model.state_dict())
    differences=[]
    for count in (64,256,1024,4096):
        algorithm.populationPertubationPopulationSize=count
        model.load_state_dict(original)
        pt.manual_seed(99)
        algorithm.trainOrTestModel(model,True,x,y,None,None)
        reference=[p.clone() for p in model.parameters()]
        reference_rng=pt.get_rng_state()
        model.load_state_dict(original)
        pt.manual_seed(99)
        population_step(model,algorithm,x,y,chunk_size=64)
        for actual, expected in zip(model.parameters(),reference):
            pt.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        assert pt.equal(reference_rng,pt.get_rng_state())
        differences.append({'population':count,'maximum_parameter_difference':max((p-q).abs().max().item() for p,q in zip(model.parameters(),reference))})
    for count in (64,256,4096):
        algorithm.populationPertubationPopulationSize=count
        if device.type=='cuda':pt.cuda.synchronize()
        start=time.perf_counter()
        population_step(model,algorithm,x,y)
        if device.type=='cuda':pt.cuda.synchronize()
        print('timing',count,time.perf_counter()-start,flush=True)
    import json
    from pathlib import Path
    Path(__file__).with_name('batched_verification.json').write_text(json.dumps({'device':str(device),'comparisons':differences},indent=2))
    print('Verified',differences,flush=True)
