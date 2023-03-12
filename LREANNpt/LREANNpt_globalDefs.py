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
LREANNpt globalDefs

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

#initialise (dependent vars);
datasetNormalise = False
datasetRepeat = False
datasetShuffle = False
datasetOrderByClass = False
dataloaderShuffle = True
dataloaderMaintainBatchSize = True
dataloaderRepeat = False
optimiserAdam = True	#initialise (dependent var)
useCustomWeightInitialisation = False	#initialise (dependent var)
useCustomBiasInitialisation = False	#initialise (dependent var)

trainLocal = False	#initialise (dependent var)
if(useAlgorithmAUANN):
	trainLocal = True	#required	#local learning rule	#disable for debug (standard backprop algorithm)
	if(trainLocal):
		AUANNadjustLearningRateBasedOnNumberClasses = False
	dataloaderRepeat = True	#optional (used to test AUANN algorithm only; that there is a sufficient number of adjacent/contiguous same-class samples in the dataset)
	if(dataloaderRepeat):
		dataloaderRepeatSize = 10
	AUANNtrainDiscordantClassExperiences = False	#not yet coded (require discordant crossEntropyLoss function)
	debugOnlyTrainLastLayer = False

if(useCustomWeightInitialisation):
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations
	
#initialise (dependent vars);
datasetReplaceNoneValues = False
datasetNormaliseClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False

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
	
#datasetName = 'tabular-benchmark'
#datasetName = 'blog-feedback'
#datasetName = 'titanic'
#datasetName = 'red-wine'
#datasetName = 'breast-cancer-wisconsin'
#datasetName = 'diabetes-readmission'
datasetName = 'new-thyroid'
if(datasetName == 'tabular-benchmark'):
	datasetNameFull = 'inria-soda/tabular-benchmark'
	classFieldName = 'class'
	trainFileName = 'clf_cat/albert.csv'
	testFileName = 'clf_cat/albert.csv'
	datasetNormalise = True
elif(datasetName == 'blog-feedback'):
	datasetNameFull = 'wwydmanski/blog-feedback'
	classFieldName = 'target'
	trainFileName = 'train.csv'
	testFileName = 'test.csv'
	datasetNormalise = True
	datasetNormaliseClassValues = True	#int: not contiguous	#CHECKTHIS
elif(datasetName == 'titanic'):
	datasetNameFull = 'victor/titanic'
	classFieldName = '2urvived'
	trainFileName = 'train_and_test2.csv'	#train
	testFileName = 'train_and_test2.csv'	#test
	datasetReplaceNoneValues = True
	datasetNormalise = True
elif(datasetName == 'red-wine'):
	datasetNameFull = 'lvwerra/red-wine'
	classFieldName = 'quality'
	trainFileName = 'winequality-red.csv'
	testFileName = 'winequality-red.csv'
	datasetNormaliseClassValues = True	#int: not start at 0
	datasetNormalise = True
elif(datasetName == 'breast-cancer-wisconsin'):
	datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
	classFieldName = 'diagnosis'
	trainFileName = 'breast_cancer.csv'
	testFileName = 'breast_cancer.csv'
	datasetReplaceNoneValues = True
	datasetNormaliseClassValues = True	#string: B/M	#requires conversion of target string B/M to int
	datasetNormalise = True
elif(datasetName == 'diabetes-readmission'):
	datasetNameFull = 'imodels/diabetes-readmission'
	classFieldName = 'readmitted'
	trainFileName = 'train.csv'
	testFileName = 'test.csv'	
	datasetNormalise = True
elif(datasetName == 'new-thyroid'):
	classFieldName = 'class'
	trainFileName = 'new-thyroid.csv'
	testFileName = 'new-thyroid.csv'
	datasetLocalFile = True	
	datasetNormalise = True
	datasetNormaliseClassValues = True
	trainNumberOfEpochs = 100
	batchSize = 100	#emulate VICRegANNtf
	numberOfLayers = 4
	hiddenLayerSize = 15	#5
	datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
	if(datasetRepeat):
		datasetRepeatSize = 10
	dataloaderRepeat = True
#elif ...

dataloaderRepeatSampler = False	#initialise (dependent var)
if(dataloaderRepeat):
	dataloaderRepeatSize = 10	#number of repetitions
	dataloaderRepeatSampler = True
	dataloaderRepeatLoop = False	#not supported
	if(dataloaderRepeatSampler):
		dataloaderRepeatSamplerCustom = False	#no tqdm visualisation

if(datasetNormalise):
	datasetNormaliseMinMax = True	#normalise between 0.0 and 1.0
	datasetNormaliseStdAvg = False	#normalise based on std and mean (~-1.0 to 1.0)


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

dataDrive = '/large/source/ANNpython/LREANNpt/'	#'/datasets/'
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
