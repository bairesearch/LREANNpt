"""LREANNpt_AUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNpt_main.py

# Usage:
see LREANNpt_main.py

# Description:
LREANNpt associative (wrt previous experience class) update artificial neural network (AUANN)

"""

from LREANNpt_globalDefs import *
import LREANNpt_AUANNmodel
import LREANNpt_data

def createModel(dataset):
	datasetSize = LREANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = LREANNpt_data.countNumberFeatures(dataset)
	numberOfClasses = LREANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = LREANNpt_AUANNmodel.AUANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses
	)
	model = LREANNpt_AUANNmodel.AUANNmodel(config)
	return model
	
