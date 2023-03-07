"""LREANNpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNpt_main.py

# Usage:
see LREANNpt_main.py

# Description:
LREANNpt data 

"""


import torch
from datasets import load_dataset
from LREANNpt_globalDefs import *

def loadDataset():
	dataset = load_dataset(datasetNameFull, data_files={"train":trainFileName, "test":testFileName})
	if(datasetShuffle):
		dataset = shuffleDataset(dataset)
	elif(datasetOrderByClass):
		dataset = orderDatasetByClass(dataset)
	return dataset

def shuffleDataset(dataset):
	datasetSize = getDatasetSize(dataset)
	dataset = dataset.shuffle()
	return dataset
	
def orderDatasetByClass(dataset):
	dataset = dataset.sort(classFieldName)
	return dataset
			
def countNumberClasses(dataset, printSize=True):
	numberOfClassSamples = {}
	datasetSize = getDatasetSize(dataset)
	numberOfClasses = 0
	for i in range(datasetSize):
		row = dataset[i]
		target = int(row[classFieldName])
		if(target in numberOfClassSamples):
			numberOfClassSamples[target] = numberOfClassSamples[target] + 1
		else:
			numberOfClassSamples[target] = 0
		#print("target = ", target)
		if(target > numberOfClasses):
			numberOfClasses = target
	numberOfClasses = numberOfClasses+1
	if(printSize):
		print("numberOfClasses = ", numberOfClasses)
	return numberOfClasses, numberOfClassSamples

def countNumberFeatures(dataset, printSize=True):
	numberOfFeatures = len(dataset.features)-1	#-1 to ignore class targets
	if(printSize):
		print("numberOfFeatures = ", numberOfFeatures)
	return numberOfFeatures
	
def getDatasetSize(dataset, printSize=False):
	datasetSize = dataset.num_rows
	if(printSize):
		print("datasetSize = ", datasetSize)
	return datasetSize
	
def createDataLoader(dataset):
	dataLoaderDataset = DataloaderDatasetInternet(dataset)	
	loader = torch.utils.data.DataLoader(dataLoaderDataset, batch_size=batchSize, shuffle=True)	#shuffle not supported by DataloaderDatasetHDD

	#loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
	return loader

class DataloaderDatasetInternet(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.datasetSize = getDatasetSize(dataset)
		self.datasetIterator = iter(dataset)
			
	def __len__(self):
		return self.datasetSize

	def __getitem__(self, i):
		document = next(self.datasetIterator)
		documentList = list(document.values())
		x = documentList[0:-1]
		y = documentList[-1]
		x = torch.Tensor(x).float()
		batchSample = (x, y)
		return batchSample
		
