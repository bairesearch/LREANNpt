"""ANNpt_linearSublayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt linear sublayers

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *

class LinearSegregated(nn.Module):
	def __init__(self, in_features, out_features, number_sublayers):
		super().__init__()
		self.segregatedLinear = nn.Conv1d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=1, groups=number_sublayers)
		self.number_sublayers = number_sublayers
		
	def forward(self, x):
		#x.shape = batch_size, number_sublayers, in_features
		x = x.view(x.shape[0], x.shape[1]*x.shape[2], 1)
		x = self.segregatedLinear(x)
		x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers)
		#x.shape = batch_size, number_sublayers, out_features
		return x

def generateLinearLayer(self, layerIndex, config):
	if(layerIndex == 0):
		in_features = config.inputLayerSize
	else:
		if(useLinearSublayers):
			in_features = config.hiddenLayerSize*config.linearSublayersNumber
		else:
			in_features = config.hiddenLayerSize
	if(layerIndex == config.numberOfLayers-1):
		out_features = config.outputLayerSize
	else:
		out_features = config.hiddenLayerSize

	if(getUseLinearSublayers(self, layerIndex)):
		linear = LinearSegregated(in_features=in_features, out_features=out_features, number_sublayers=config.linearSublayersNumber)
	else:
		linear = nn.Linear(in_features=in_features, out_features=out_features)

	weightsSetLayer(self, layerIndex, linear)

	return linear

def generateActivationLayer(self, layerIndex, config):
	if(usePositiveWeights):
		if(getUseLinearSublayers(self, layerIndex)):
			activation = nn.Softmax(dim=1)
		else:
			activation = nn.Softmax(dim=1)
	else:
		if(getUseLinearSublayers(self, layerIndex)):
			activation = nn.ReLU()
		else:
			activation = nn.ReLU()		
	return activation

def executeLinearLayer(self, layerIndex, x, linear):
	weightsFixLayer(self, layerIndex, linear)	#otherwise need to constrain backprop weight update function to never set weights below 0
	if(getUseLinearSublayers(self, layerIndex)):
		#perform computation for each sublayer independently
		x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		x = linear(x)
	else:
		x = linear(x)
	return x

def executeActivationLayer(self, layerIndex, x, softmax):
	if(getUseLinearSublayers(self, layerIndex)):
		xSublayerList = []
		for sublayerIndex in range(self.config.linearSublayersNumber):
			xSublayer = x[:, sublayerIndex]
			xSublayer = softmax(xSublayer)	#or torch.nn.functional.softmax(xSublayer)
			xSublayerList.append(xSublayer)
		x = torch.concat(xSublayerList, dim=1)
	else:
		x = softmax(x)
	return x

def getUseLinearSublayers(self, layerIndex):
	result = False
	if(useLinearSublayers):
		if(layerIndex != self.config.numberOfLayers-1):	#final layer does not useLinearSublayers
			result = True
	return result

def weightsSetLayer(self, layerIndex, linear):
	weightsSetPositiveLayer(self, layerIndex, linear)
	if(useCustomWeightInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.normal_(linear.segregatedLinear.weight, mean=Wmean, std=WstdDev)
		else:
			nn.init.normal_(linear.weight, mean=Wmean, std=WstdDev)
	if(useCustomBiasInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.constant_(linear.segregatedLinear.bias, 0)
		else:
			nn.init.constant_(linear.bias, 0)

def weightsFixLayer(self, layerIndex, linear):
	weightsSetPositiveLayer(self, layerIndex, linear)
			
def weightsSetPositiveLayer(self, layerIndex, linear):
	if(usePositiveWeights):
		if(not usePositiveWeightsClampModel):
			if(getUseLinearSublayers(self, layerIndex)):
				weights = linear.segregatedLinear.weight #only positive weights allowed
				weights = torch.abs(weights)
				linear.segregatedLinear.weight = torch.nn.Parameter(weights)
			else:
				weights = linear.weight #only positive weights allowed
				weights = torch.abs(weights)
				linear.weight = torch.nn.Parameter(weights)
		
def weightsSetPositiveModel(self):
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)
