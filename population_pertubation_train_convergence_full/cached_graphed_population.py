"""Experimental CUDA-graph dispatch for the already-verified batched estimator.

A single-chunk population reuses its original noise for the weighted sum.
Chunk RNG order, centring and arithmetic are unchanged. RNG rewind happens on
host between graphs; all custom generators are registered with their graphs.
"""
import torch as pt
from fast_population import PopulationStep

class CachedGraphedPopulationStep(PopulationStep):
    @pt.no_grad()
    def _build(self,x,y):
        model=self.model;algorithm=self.algorithm
        assert algorithm.usePopulationPertubation and not list(model.buffers())
        self.x=x.clone();self.y=y.clone()
        self.names,self.parameters=zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.numel()])
        self.count=algorithm.populationPertubationPopulationSize;self.sigma=algorithm.populationPertubationSigma
        self.cache_noise=self.count<=self.chunk_size
        self.reward=pt.zeros(self.count,device=x.device);self.weights=pt.zeros_like(self.reward)
        self.updates=[pt.zeros_like(p) for p in self.parameters]
        self.scale=pt.zeros((),device=x.device,dtype=pt.float64);self.last_scale=None
        self.reward_graphs={};self.update_graphs={};self.part_rewards={};self.part_weights={}
        sizes=sorted({min(self.chunk_size,self.count-i) for i in range(0,self.count,self.chunk_size)})
        stream=pt.cuda.Stream();stream.wait_stream(pt.cuda.current_stream())
        rng=self.generator.get_state()
        flag=getattr(model,'_populationPertubationEvaluating',False);had=hasattr(model,'_populationPertubationEvaluating')
        model._populationPertubationEvaluating=True
        def loss_for_params(params):return pt.func.functional_call(model,params,(True,self.x,self.y,None,None))[0]
        def rewards(size):
            noise=self.noises(self.parameters,size)
            if self.cache_noise:self.cached_noise=noise
            candidates={n:p.detach().unsqueeze(0)+self.sigma*e for n,p,e in zip(self.names,self.parameters,noise)}
            return -pt.vmap(loss_for_params)(candidates)
        def updates(size):
            noise=self.cached_noise if self.cache_noise else self.noises(self.parameters,size)
            for u,e in zip(self.updates,noise):u.add_(pt.einsum('n,n...->...',self.part_weights[size],e))
        def centre():
            self.weights.copy_(((self.reward.double()-self.reward.double().mean())*self.scale).float())
            for u in self.updates:u.zero_()
            self.reward_finite=pt.isfinite(self.reward).all()
        def finish():
            for p,u in zip(self.parameters,self.updates):p.add_(u)
            self.loss=model(True,self.x,self.y,None,None)[0]
            self.parameter_finite=pt.stack([pt.isfinite(p).all() for p in self.parameters]).all()
        try:
            with pt.cuda.stream(stream):
                for size in sizes:
                    self.part_weights[size]=pt.zeros(size,device=x.device)
                    for _ in range(3):rewards(size);updates(size)
                for _ in range(3):centre();finish()
            pt.cuda.current_stream().wait_stream(stream)
            pt.cuda.synchronize()
            for size in sizes:
                g=pt.cuda.CUDAGraph();g.register_generator_state(self.generator)
                with pt.cuda.graph(g,stream=stream):self.part_rewards[size]=rewards(size)
                self.reward_graphs[size]=g
                g=pt.cuda.CUDAGraph();g.register_generator_state(self.generator)
                with pt.cuda.graph(g,stream=stream):updates(size)
                self.update_graphs[size]=g
            self.centre_graph=pt.cuda.CUDAGraph()
            with pt.cuda.graph(self.centre_graph,stream=stream):centre()
            self.finish_graph=pt.cuda.CUDAGraph()
            with pt.cuda.graph(self.finish_graph,stream=stream):finish()
            pt.cuda.synchronize();self.generator.set_state(rng)
        finally:
            if had:model._populationPertubationEvaluating=flag
            else:del model._populationPertubationEvaluating
        self.built=True

    @pt.no_grad()
    def __call__(self,x,y):
        if not getattr(self,'built',False):self._build(x,y)
        assert self.count==self.algorithm.populationPertubationPopulationSize and self.sigma==self.algorithm.populationPertubationSigma
        self.x.copy_(x);self.y.copy_(y)
        scale=self.algorithm.populationPertubationLearningRate/(self.count*self.sigma)
        if scale!=self.last_scale:self.scale.fill_(scale);self.last_scale=scale
        rng=self.generator.get_state()
        for start in range(0,self.count,self.chunk_size):
            size=min(self.chunk_size,self.count-start)
            self.reward_graphs[size].replay();self.reward[start:start+size].copy_(self.part_rewards[size])
        self.centre_graph.replay()
        if not self.reward_finite.item():raise FloatingPointError('Nonfinite candidate reward')
        if not self.cache_noise:self.generator.set_state(rng)
        for start in range(0,self.count,self.chunk_size):
            size=min(self.chunk_size,self.count-start)
            self.part_weights[size].copy_(self.weights[start:start+size]);self.update_graphs[size].replay()
        self.finish_graph.replay()
        if not self.parameter_finite.item():raise FloatingPointError('Nonfinite population update')
        return self.loss,0
