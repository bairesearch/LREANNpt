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

#debug parameters:
printSUANNmodelProperties = True

#dataset parameters:
useTabularDataset = True
useImageDataset = False	#use CIFAR-10 dataset with CNN 

#SUANN architecture parameters:
useStochasticUpdates = True
if(useStochasticUpdates):
	#SUANN optimisation parameters:
	useEvolutionarySearch = True	#default: True (ES:EGGROLL) #orig: False (individual weight pertubations)
	if(useEvolutionarySearch):	#ES:EGGROLL
		evolutionaryPopulationSize = 64	#default: 8, 64	#number of perturbations per matrix update (N)
		evolutionaryRank = 1	#rank-r factors used for each perturbation (EGGROLL paper default)
		evolutionarySigma = 0.05	#orig: 0.05	#scale applied to rank-r perturbations before evaluation (σ)
		evolutionaryLearningRate = 0.01	#orig: 0.01 #α in Algorithm 1 (should typically match learningRateBase)
		evolutionaryNormalizeFitness = True	#subtract mean/standard deviation of fitness values before weighting
		evolutionaryPerturbAllTrainableTensors = False	#orig: False	#optional optimisation: perturb all tensors jointly per worker
		evolutionaryAntitheticSampling = True	#orig: False
		if(evolutionaryAntitheticSampling):
			evolutionaryFitnessShaping = True	#orig: False	#optional: convert antithetic pair fitness to +/-1 signals
		else:
			evolutionaryFitnessShaping = False
		evolutionaryPopulationSigmaScheduling = False	#orig: False	#optional: decay sigma across population members
	else:	#individual weight pertubations
		learningRateBase = 0.01
		#fractionTensorParametersPertubatedPerIteration = 1.0
	trainLocal = True	#mandatory: True	#execute training at each layer (LREANNpt_SUANN training code), do not execute training at final layer only (ANNpt_main training code)
else:
	trainLocal = False	#default: False #disable for debug/benchmark against standard full layer backprop
	useEvolutionarySearch = False

#skip layer parameters:
supportSkipLayers = False #default: False 	#fully connected skip layer network
supportSkipLayersResidual = False	#default: False	#direct residual connections

if(useImageDataset):
	useCNNlayers = True		#mandatory:True
elif(useTabularDataset):
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

trainNumberOfEpochsLow = False	#default: False
trainNumberOfEpochsHigh = False	#default: False	#orig: True	#use ~4x more epochs to train

#activation function parameters:
activationFunctionTypeForward = "relu"

#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

if(useTabularDataset):
	datasetType = "useTabularDataset"
elif(useImageDataset):
	datasetType = "useImageDataset"

#data storage parameters:
workingDrive = '/large/source/ANNpython/LREANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelSUANN'
