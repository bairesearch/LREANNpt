"""LREANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNpt_main.py

# Usage:
see LREANNpt_main.py

# Description:
LIANNpt globalDefs

"""

import torch
useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	torch.set_printoptions(profile="full")

torch.autograd.set_detect_anomaly(True)

useAlgorithmAUANN = True

stateTrainDataset = True
stateTestDataset = True


trainLocal = False
datasetShuffle = False
datasetOrderByClass = False
if(useAlgorithmAUANN):
	trainLocal = True	#required	#local learning rule	#disable for debug (standard backprop algorithm)
	datasetShuffle = True	#recommended
	if(not datasetShuffle):
		datasetOrderByClass = True	#optional (used to test AUANN algorithm only; that there is a sufficient number of adjacent/contiguous same-class samples in the dataset)
	AUANNtrainDiscordantClassExperiences = False	#not yet coded (require discordant crossEntropyLoss function)
	AUANNadjustLearningRateBasedOnNumberClasses = False
	debugOnlyTrainLastLayer = False

datasetReplaceNoneValues = False
datasetNormaliseClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)

#datasetName = 'tabular-benchmark'
#datasetName = 'blog-feedback'
#datasetName = 'titanic'
#datasetName = 'red-wine'
#datasetName = 'breast-cancer-wisconsin'
datasetName = 'diabetes-readmission'
if(datasetName == 'tabular-benchmark'):
	datasetNameFull = 'inria-soda/tabular-benchmark'
	classFieldName = 'class'
	trainFileName = 'clf_cat/albert.csv'
	testFileName = 'clf_cat/albert.csv'
elif(datasetName == 'blog-feedback'):
	datasetNameFull = 'wwydmanski/blog-feedback'
	classFieldName = 'target'
	trainFileName = 'train.csv'
	testFileName = 'test.csv'
	datasetNormaliseClassValues = True	#int: not contiguous	#CHECKTHIS
elif(datasetName == 'titanic'):
	datasetNameFull = 'victor/titanic'
	classFieldName = '2urvived'
	trainFileName = 'train_and_test2.csv'	#train
	testFileName = 'train_and_test2.csv'	#test
	datasetReplaceNoneValues = True
elif(datasetName == 'red-wine'):
	datasetNameFull = 'lvwerra/red-wine'
	classFieldName = 'quality'
	trainFileName = 'winequality-red.csv'
	testFileName = 'winequality-red.csv'
	datasetNormaliseClassValues = True	#int: not start at 0
elif(datasetName == 'breast-cancer-wisconsin'):
	datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
	classFieldName = 'diagnosis'
	trainFileName = 'breast_cancer.csv'
	testFileName = 'breast_cancer.csv'
	datasetReplaceNoneValues = True
	datasetNormaliseClassValues = True	#string: B/M	#requires conversion of target string B/M to int
elif(datasetName == 'diabetes-readmission'):
	datasetNameFull = 'imodels/diabetes-readmission'
	classFieldName = 'readmitted'
	trainFileName = 'train.csv'
	testFileName = 'test.csv'	
#elif ...

debugSmallNetwork = False
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1	#default: 10	#number of epochs to train
else:
	batchSize = 64
	numberOfLayers = 4
	hiddenLayerSize = 100
	trainNumberOfEpochs = 10	#default: 10	#number of epochs to train

printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10


useInbuiltCrossEntropyLossFunction = True	#required
usePositiveWeights = False	#experimental only
if(usePositiveWeights):
	usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required


learningRate = 0.005	#0.0001	#0.005	#0.0001

useLinearSublayers = False	#use multiple independent sublayers per linear layer
if(useLinearSublayers):
	linearSublayersNumber = 10
else:
	linearSublayersNumber = 1
	

relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE

modelSaveNumberOfBatches = 100	#resave model after x training batches

dataDrive = '/datasets/'
workingDrive = '/large/source/ANNpython/LREANNpt/'

dataFolderName = 'data'
modelFolderName = 'model'
if(relativeFolderLocations):
	dataPathName = dataFolderName
	modelPathName = modelFolderName
else:	
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
modelPathNameBase = modelPathName
modelName = 'modelLREANN'
modelPathNameFull = getModelPathNameFull(modelPathNameBase, modelName)
	
def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = torch.cuda.get_device_properties(0).total_memory
	memory_reserved = torch.cuda.memory_reserved(0)
	memory_allocated = torch.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()
