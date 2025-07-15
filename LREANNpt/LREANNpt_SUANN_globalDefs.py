"""LREANNpt_SUANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LREANNpt SUANN globalDefs

"""

#SUANN architecture parameters:
useStochasticUpdates = True
if(useStochasticUpdates):
	trainLocal = True	#mandatory: True	#execute training at each layer (LREANNpt_SUANN training code), do not execute training at final layer only (ANNpt_main training code)
else:
	trainLocal = False	#default: False #disable for debug/benchmark against standard full layer backprop
supportSkipLayers = False #optional	#fully connected skip layer network
supportSkipLayersResidual = False	#optional	#direct residual connections

numberTensorParametersPertubatedPerIteration = 100	#depends on average number of elements per parameter tensor

#dataset parameters:
useImageDataset = False	#use CIFAR-10 dataset with CNN 
if(useImageDataset):
	useTabularDataset = False
	useCNNlayers = True		#mandatory:True
else:
	useTabularDataset = True
	useCNNlayers = False	 #default:False	#optional	#enforce different connection sparsity across layers to learn unique features with greedy training	#use 2D CNN instead of linear layers

#CNN parameters:
if(useImageDataset):
	#create CNN architecture, where network size converges by a factor of ~4 (or 2*2) per layer and number of channels increases by the same factor
	CNNkernelSize = 3
	CNNstride = 1
	CNNpadding = "same"
	useCNNlayers2D = True
	CNNinputWidthDivisor = 2
	CNNinputSpaceDivisor = CNNinputWidthDivisor*CNNinputWidthDivisor
	CNNinputPadding = False
	CNNmaxInputPadding = False	#pad input with zeros such that CNN is applied to every layer
	debugCNN = False
	if(CNNstride == 1):
		CNNmaxPool = True
		#assert not supportSkipLayers, "supportSkipLayers not currently supported with CNNstride == 1 and CNNmaxPool"
	elif(CNNstride == 2):
		CNNmaxPool = False
		assert CNNkernelSize==2
	else:
		print("error: CNNstride>2 not currently supported")
	CNNbatchNorm = True
else:
	CNNmaxPool = False
	CNNbatchNorm = False

#learning rate parameters;
learningRateBase = 0.0001

#activation function parameters:
activationFunctionTypeForward = "relu"

#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

#data storage parameters:
workingDrive = '/large/source/ANNpython/LREANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelSUANN'
