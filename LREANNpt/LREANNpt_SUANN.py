"""LREANNpt_SUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LREANNpt stochastic update artificial neural network (SUANN)

"""


import torch as pt
from contextlib import contextmanager
from typing import Callable, Dict, Tuple
from torchsummary import summary

from ANNpt_globalDefs import *
import LREANNpt_SUANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = LREANNpt_SUANNmodel.SUANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSize,
		CNNhiddenLayerSize = CNNhiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = LREANNpt_SUANNmodel.SUANNmodel(config)
	
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
def trainOrTestModel(model, trainOrTest, x, y, optim, l):
	if(trainOrTest and useStochasticUpdates):
		#print("trainOrTestModel:trainOrTest")
		with pt.no_grad():
			for name, p in model.named_parameters():	#for every parameter tensor in model	#FUTURE; ensure to iterate in order of layer order
				#print("parameter tensor name = ", name)
				numberTensorElements = p.numel()
				#print("numberTensorParameters = ", numberTensorParameters)
				for idx in range(numberTensorElements):
					#print("random weight selection index = ", i)
					loss1, accuracy1 = model(trainOrTest, x, y, optim, l)
					idx, backup = perturb_once(model, name, p, idx)
					loss2, accuracy2 = model(trainOrTest, x, y, optim, l)
					update_weight(model, idx, backup, name, p, idx, loss1, loss2)
			loss, accuracy = model(trainOrTest, x, y, optim, l)	#final pass for loss/acc calc
	else:
		loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy

def pertubation_function(x): 
	noise = learningRateBase	#CHECKTHIS
	#noise = learningRateBase * x.sign() * x.abs()	 # e.g. add 0.01% of magnitude, sign-preserving
	return noise

def perturb_once(model: pt.nn.Module, name: str, p, idx):
	"""
	Add `pertubation_function(old_val)` to *one* randomly chosen element (idx) of the parameter tensor
	Returns:  idx (of randomly chosen element), backup (of original value before pertubation)
	"""
	info: PerturbInfo = {}
	backup = (p.detach().clone(), idx)	#Tuple[pt.Tensor, int]
	flat = p.view(-1)
	flat[idx] += pertubation_function(flat[idx])
	return idx, backup
	
def update_weight(model: pt.nn.Module, idx: int, backup: Tuple[pt.Tensor, int], name: str, p, e, loss1, loss2):
	lossDiff = loss1 - loss2	#positive = a good change, negative = a bad change
	p.copy_(backup[0])  # restore exact original bytes
	flat = p.view(-1)
	update = lossDiff*100.0	#*100.0 to ensure average update is approximately learningRateBase	#CHECKTHIS
	#print("lossDiff = ", lossDiff)
	#print("update = ", update)
	flat[e] += update		 # <-- permanent change

def rand_index(t: pt.Tensor) -> int:
	"""Return a random flat index for tensor `t`."""
	numel = t.numel()
	return pt.randint(numel, (), device=t.device).item()
