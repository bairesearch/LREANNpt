"""Load frozen SUANN modules with per-process, explicitly recorded overrides."""
import importlib, re, sys, types
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source/LREANNpt'

def load_modules(method,dataset):
    sys.path.insert(0,str(SOURCE))
    config=types.ModuleType('LREANNpt_SUANN_globalDefs')
    source=(SOURCE/'LREANNpt_SUANN_globalDefs.py').read_text()
    source=re.sub(r'(?m)^useStochasticUpdates = .*$', 'useStochasticUpdates = True', source)
    for key,value in [('useIndividualPertubation',False),('usePopulationPertubation',True),('useEvolutionarySearch',False)]:
        source=re.sub(r'(?m)^\t'+key+r' = .*$',f'\t{key} = {value}',source)
    source=re.sub(r'(?m)^\t\tpopulationPertubationOptimiseTrainingIterations = .*$', '\t\tpopulationPertubationOptimiseTrainingIterations = True',source)
    exec(compile(source,str(SOURCE/'LREANNpt_SUANN_globalDefs.py'),'exec'),vars(config))
    #Keep the same controller settings for Adam; its updates still use backprop.
    config.useStochasticUpdates=method.startswith('population_')
    config.trainLocal=config.useStochasticUpdates
    sys.modules[config.__name__]=config
    definitions=types.ModuleType('ANNpt_globalDefs')
    source=(SOURCE/'ANNpt_globalDefs.py').read_text()
    source,count=re.subn(r"(?m)^\tdatasetName = 'titanic'.*$",f'\tdatasetName = {dataset!r}',source)
    assert count==1
    exec(compile(source,str(SOURCE/'ANNpt_globalDefs.py'),'exec'),vars(definitions))
    import torch
    definitions.device=torch.device('cuda')
    definitions.printSUANNmodelProperties=False
    sys.modules[definitions.__name__]=definitions
    algorithm=importlib.import_module('LREANNpt_SUANN')
    convergence=importlib.import_module('LREANNpt_SUANN_trainingConvergence')
    return definitions,algorithm,convergence
