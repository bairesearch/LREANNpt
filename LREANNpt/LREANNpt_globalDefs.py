"""LREANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LREANNpt globalDefs

"""

trainLocal = True	#required	#local learning rule	#disable for debug (standard backprop algorithm)
if(trainLocal):
	AUANNadjustLearningRateBasedOnNumberClasses = False
	trainIndividialSamples = True
dataloaderRepeat = True	#optional (used to test AUANN algorithm only; that there is a sufficient number of adjacent/contiguous same-class samples in the dataset)
if(dataloaderRepeat):
	dataloaderRepeatSize = 10
AUANNtrainDiscordantClassExperiences = False	#not yet coded (require discordant crossEntropyLoss function)
debugOnlyTrainLastLayer = False

workingDrive = '/large/source/ANNpython/LREANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLREANN'
