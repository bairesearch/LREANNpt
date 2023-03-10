"""LREANNpt_AUANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNpt_main.py

# Usage:
see LREANNpt_main.py

# Description:
LREANNpt associative (wrt previous experience class) update artificial neural network (AUANN) model

uses the previous experience/sample class as the exemplar (different implementation to LREANNtf_AUANN):
	if the previous experience class is concordant with current experience class, then shift local layer weights to make activations more aligned with previous experience
	if the previous experience class is discordant with current experience class, then shift local layer weights to make activations less aligned with previous experience

"""

import torch as pt
from torch import nn
from LREANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers
	
class AUANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples

class AUANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config)
			layersActivationList.append(activation)
		self.layersLinear = nn.ModuleList(layersLinearList)
		self.layersActivation = nn.ModuleList(layersActivationList)
	
		self.lossFunction = nn.CrossEntropyLoss()
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)

		if(trainLocal):
			self.previousSampleStatesLayerList = [None]*config.numberOfLayers
			self.previousSampleClass = None

	def forward(self, trainOrTest, x, y, optim):
		if(trainLocal and trainOrTest and not debugOnlyTrainLastLayer):
			loss, accuracy = self.forwardSamples(x, y, optim)
		else:
			loss, accuracy = self.forwardBatchStandard(x, y)	#standard backpropagation
		return loss, accuracy

	def forwardBatchStandard(self, x, y):
		#print("forwardBatchStandard")
		for layerIndex in range(self.config.numberOfLayers):
			if(debugOnlyTrainLastLayer and (layerIndex == self.config.numberOfLayers-1)):
				x = x.detach()
			x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
			if(layerIndex != self.config.numberOfLayers-1):
				x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		return loss, accuracy
		
	def forwardSamples(self, x, y, optim):
		maxSampleIndex = x.shape[0]	#not guaranteed to be batchSize (for last batch in dataset)
		lossAverage = 0.0
		accuracyAverage = 0.0
		self.previousSampleClass = None
		for sampleIndex in range(batchSize):
			if(sampleIndex < maxSampleIndex):
				loss, accuracy = self.forwardSample(sampleIndex, x[sampleIndex].unsqueeze(0), y[sampleIndex].unsqueeze(0), optim)
				self.previousSampleClass = y[sampleIndex]
				lossAverage = lossAverage + loss
				accuracyAverage = accuracyAverage + accuracy
		lossAverage = lossAverage/maxSampleIndex
		accuracyAverage = accuracyAverage/maxSampleIndex
		return lossAverage, accuracyAverage
		
	def forwardSample(self, sampleIndex, x, y, optim):
		for layerIndex in range(self.config.numberOfLayers):
			if(AUANNtrainDiscordantClassExperiences):
				x, loss, accuracy = self.forwardSampleLayer((self.previousSampleClass is not None), layerIndex, sampleIndex, x, y, optim)
			else:
				x, loss, accuracy = self.forwardSampleLayer((y == self.previousSampleClass), layerIndex, sampleIndex, x, y, optim)
		#will return the loss/accuracy from the last layer
		return loss, accuracy

	def forwardSampleLayer(self, trainLayerCriterion, layerIndex, sampleIndex, x, y, optim):
		if(trainLayerCriterion):
			x, loss, accuracy = self.trainSampleLayer(layerIndex, sampleIndex, x, y, optim)
		else:
			if(layerIndex == self.config.numberOfLayers-1):
				x, loss, accuracy = self.trainSampleLayer(layerIndex, sampleIndex, x, y, optim)
			else:
				x = self.propagateSampleLayer(layerIndex, x)
				loss = accuracy = None	#not used
		return x, loss, accuracy
		
	def trainSampleLayer(self, layerIndex, sampleIndex, x, y, optim):
		x = x.detach()
		
		optim = optim[sampleIndex][layerIndex]
		if(AUANNadjustLearningRateBasedOnNumberClasses):
			optim = self.normaliseLearningRate(optim, y)
		optim.zero_grad()
			
		x, loss, accuracy = self.trainSampleLayer2(layerIndex, sampleIndex, x, y)
		
		loss.backward()
		optim.step()
		
		return x, loss, accuracy
	
	def trainSampleLayer2(self, layerIndex, sampleIndex, x, y):
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		accuracy = None
		
		if(layerIndex == self.config.numberOfLayers-1):
			loss = self.lossFunction(x, y)
			accuracy = self.accuracyFunction(x, y)
			accuracy = accuracy.detach().cpu().numpy()
		else:
			x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
			if(y == self.previousSampleClass):
				loss = self.lossFunction(x, self.previousSampleStatesLayerList[layerIndex])
			else:
				loss = 1/(self.lossFunction(x, self.previousSampleStatesLayerList[layerIndex]))	#CHECKTHIS - determine discordant class loss function
				#print("loss = ", loss)
			previousSampleStatesLayer = x.detach().clone()
			self.previousSampleStatesLayerList[layerIndex] = previousSampleStatesLayer
		
		return x, loss, accuracy

	def propagateSampleLayer(self, layerIndex, x):
		x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
		x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
		previousSampleStatesLayer = x.detach().clone()
		self.previousSampleStatesLayerList[layerIndex] = previousSampleStatesLayer
		return x

	def normaliseLearningRate(self, optim, y):
		classTarget = y[0].detach().cpu().numpy().item() 
		learningRateBias = (self.config.datasetSize/self.config.numberOfClasses)/self.config.numberOfClassSamples[classTarget]
		learningRateClass = learningRate*learningRateBias
		#print("learningRateBias = ", learningRateBias)
		#print("learningRateClass = ", learningRateClass)
		optim = self.setLearningRate(optim, learningRateClass)
		return optim
	
	def setLearningRate(self, optim, learningRate):
		#optim.param_groups[0]['lr'] = learningRate
		for g in optim.param_groups:
			g['lr'] = learningRate
		return optim

