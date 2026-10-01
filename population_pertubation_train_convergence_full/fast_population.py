"""Benchmark-only batched iid Gaussian population estimator.

Same full-model noise distribution, reward centring, scale and SUANN forward as
production. Direct batched RNG is used from the first update in this full-data
experiment. The separate generator is fully checkpointed.
"""
import torch as pt

class PopulationStep:
    def __init__(self, model, algorithm, seed, chunk_size):
        self.model=model; self.algorithm=algorithm; self.chunk_size=chunk_size
        self.generator=pt.Generator(device=next(model.parameters()).device).manual_seed(seed)

    def noises(self, parameters, size):
        return [pt.randn((size,*p.shape),device=p.device,dtype=p.dtype,generator=self.generator) for p in parameters]

    @pt.no_grad()
    def __call__(self,x,y,provided_noise=None):
        model=self.model; algorithm=self.algorithm
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.numel()]
        names,parameters=zip(*named)
        originals=[p.detach().clone() for p in parameters]
        count=algorithm.populationPertubationPopulationSize
        sigma=algorithm.populationPertubationSigma
        rate=algorithm.populationPertubationLearningRate
        rng=self.generator.get_state()
        rewards=[]
        had=hasattr(model,'_populationPertubationEvaluating')
        flag=getattr(model,'_populationPertubationEvaluating',False)
        def loss_for_params(params):
            return pt.func.functional_call(model,params,(True,x,y,None,None))[0]
        def get_noise(start,size):
            if provided_noise is not None:return [e[start:start+size] for e in provided_noise]
            return self.noises(parameters,size)
        try:
            model._populationPertubationEvaluating=True
            for start in range(0,count,self.chunk_size):
                noise=get_noise(start,min(self.chunk_size,count-start))
                candidates={n:o.unsqueeze(0)+sigma*e for n,o,e in zip(names,originals,noise)}
                rewards.append(-pt.vmap(loss_for_params)(candidates))
            reward=pt.cat(rewards)
            if not pt.isfinite(reward).all().item():raise FloatingPointError('Nonfinite candidate reward')
            weights=((reward.double()-reward.double().mean())*(rate/(count*sigma))).float()
            self.generator.set_state(rng)
            updates=[pt.zeros_like(p) for p in parameters]
            for start in range(0,count,self.chunk_size):
                size=min(self.chunk_size,count-start)
                noise=get_noise(start,size)
                for update,epsilon in zip(updates,noise):
                    update.add_(pt.einsum('n,n...->...',weights[start:start+size],epsilon))
            candidates=[o+u for o,u in zip(originals,updates)]
            if not all(pt.isfinite(p).all().item() for p in candidates):raise FloatingPointError('Nonfinite population update')
            for parameter,candidate in zip(parameters,candidates):parameter.copy_(candidate)
        finally:
            if had:model._populationPertubationEvaluating=flag
            else:del model._populationPertubationEvaluating
        return model(True,x,y,None,None)
