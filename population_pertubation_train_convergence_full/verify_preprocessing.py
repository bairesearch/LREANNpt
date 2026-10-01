"""Verify streamed float32 preprocessing against the patched production functions."""
from pathlib import Path
import json, hashlib
from unittest.mock import patch
import numpy as np
from datasets import Dataset, disable_progress_bars
from model_setup import load_modules
HERE=Path(__file__).resolve().parent
disable_progress_bars()
_,algorithm,_=load_modules('population_64','iris')
production=algorithm.ANNpt_data
#Includes a training-constant column that differs at test time, held-out extrema,
#negative/missing-value encodings, and a very small but nonzero feature range.
x=np.array([[2,3,-1,1e-9],[4,3,0,2e-9],[6,3,1,3e-9],[12,9,-1,8e-9]],dtype=np.float32)
y=np.array([0,1,0,1]);train=np.array([0,1,2]);names=['a','constant','category','small']
def dataset(values,labels):return Dataset.from_dict({**{c:values[:,i] for i,c in enumerate(names)},'target':labels})
with patch.multiple(production,classFieldName='target',datasetNormaliseMinMax=True,datasetNormaliseStdAvg=False,datasetCorrectMissingValues=False):
    stats=production.calculateNormalisationStatistics(dataset(x[train],y[train]))
    result=production.normaliseDataset(dataset(x,y),stats)
    observed=np.stack([np.asarray(result[c],dtype=np.float32) for c in names],axis=1)
minimum=x[train].min(0);maximum=x[train].max(0)
scales=np.array([0. if float(hi)==float(lo) else 1./(float(hi)-float(lo)+1e-8) for lo,hi in zip(minimum,maximum)],dtype=np.float32)
expected=(x-minimum)*np.where(scales==0,np.float32(1),scales)
np.testing.assert_array_equal(observed,expected)
raw=Dataset.from_dict({'category':['yes','no','unseen'],'row':[0,1,2]})
converted=production.convertCategoricalFieldValues(raw,'category',dataType=float,fieldIndexDict={'no':0,'yes':1},unknownValue=-1)
np.testing.assert_array_equal(np.asarray(converted['category'],dtype=np.float32),[1,0,-1])
record={'production_normalisation_bitwise_equal':True,'training_constant_and_heldout_extrema_verified':True,
 'training_category_codes_and_unknown_minus_one_verified':True,'normalisation_statistics':stats,
 'source_sha256':hashlib.sha256((HERE/'source/LREANNpt/ANNpt_data.py').read_bytes()).hexdigest()}
(HERE/'verification/preprocessing_equivalence.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
