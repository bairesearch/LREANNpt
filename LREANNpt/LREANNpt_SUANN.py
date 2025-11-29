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
import math
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
	if(printSUANNmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t datasetType = ", datasetType)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t hiddenLayerSize = ", hiddenLayerSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t trainLocal = ", trainLocal)
		print("\t useStochasticUpdates = ", useStochasticUpdates)
		print("\t useEvolutionarySearch = ", useEvolutionarySearch)
		if(useEvolutionarySearch):
			print("\t\t evolutionaryPopulationSize = ", evolutionaryPopulationSize)
			print("\t\t evolutionaryRank = ", evolutionaryRank)
			print("\t\t evolutionarySigma = ", evolutionarySigma)
			print("\t\t evolutionaryLearningRate = ", evolutionaryLearningRate)
			print("\t\t evolutionaryNormalizeFitness = ", evolutionaryNormalizeFitness)
			print("\t\t evolutionaryPerturbAllTrainableTensors = ", evolutionaryPerturbAllTrainableTensors)
			print("\t\t evolutionaryAntitheticSampling = ", evolutionaryAntitheticSampling)
			print("\t\t evolutionaryFitnessShaping = ", evolutionaryFitnessShaping)
			print("\t\t evolutionaryPopulationSigmaScheduling = ", evolutionaryPopulationSigmaScheduling)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t\t\t numberOfConvlayers = ", numberOfConvlayers)
			print("\t\t\t numberOfFFLayers = ", numberOfFFLayers)

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
			if(useEvolutionarySearch):
				if(evolutionaryPerturbAllTrainableTensors):
					loss, accuracy = execute_evolutionary_search_all_tensors(model, trainOrTest, x, y, optim, l)
				else:
					loss, accuracy = execute_evolutionary_search_single_parameter(model, trainOrTest, x, y, optim, l)
			else:
				loss, accuracy = execute_stochastic_search(model, trainOrTest, x, y, optim, l)
	else:
		loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy

#computationally expensive;
def execute_stochastic_search(model, trainOrTest, x, y, optim, l):
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

def execute_evolutionary_search_single_parameter(model, trainOrTest, x, y, optim, l):
	for name, parameter in model.named_parameters():
		if(not parameter.requires_grad):
			continue
		apply_evolutionary_update_to_parameter(model, trainOrTest, x, y, optim, l, parameter)
	loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy

def apply_evolutionary_update_to_parameter(model, trainOrTest, x, y, optim, l, parameterTensor):
	if(parameterTensor.numel() == 0):
		return
	if(evolutionaryPopulationSize <= 0):
		return
	rows, cols = reshape_parameter_to_matrix(parameterTensor)
	perturbationFactors = []
	fitnessValues = []
	rankValue = max(evolutionaryRank, 1)
	rankScale = 1.0 / math.sqrt(rankValue)
	signGroups = generate_antithetic_sign_groups(evolutionaryPopulationSize)
	signGroupSizes = [len(group) for group in signGroups]
	memberIndex = 0
	for signGroup in signGroups:
		A = pt.randn((rows, rankValue), device=parameterTensor.device, dtype=parameterTensor.dtype)
		B = pt.randn((cols, rankValue), device=parameterTensor.device, dtype=parameterTensor.dtype)
		for sign in signGroup:
			currentSigma = get_sigma_for_member(memberIndex)
			memberIndex += 1
			perturbationMatrix = generate_low_rank_matrix(A, B, rankScale)
			perturbationDelta = currentSigma * sign * perturbationMatrix.reshape(parameterTensor.shape)
			parameterTensor.add_(perturbationDelta)
			loss, _ = model(trainOrTest, x, y, optim, l)
			parameterTensor.sub_(perturbationDelta)
			perturbationFactors.append((scale_low_rank_factor(A, sign), B))
			fitnessValues.append(-loss.item())	#minimise loss => maximise fitness
	fitnessWeights = calculate_fitness_weights(fitnessValues, parameterTensor.device, parameterTensor.dtype, signGroupSizes=signGroupSizes)
	scaledUpdate = pt.zeros((rows, cols), device=parameterTensor.device, dtype=parameterTensor.dtype)
	for weight, (A, B) in zip(fitnessWeights, perturbationFactors):
		scaledUpdate.add_(weight * generate_low_rank_matrix(A, B, rankScale))
	parameterTensor.add_((evolutionaryLearningRate / float(evolutionaryPopulationSize)) * scaledUpdate.reshape(parameterTensor.shape))

def reshape_parameter_to_matrix(parameterTensor):
	if(parameterTensor.dim() == 0):
		rows = 1
		cols = 1
	else:
		rows = parameterTensor.shape[0]
		cols = int(parameterTensor.numel() / rows)
	return rows, cols

def execute_evolutionary_search_all_tensors(model, trainOrTest, x, y, optim, l):
	if(evolutionaryPopulationSize <= 0):
		return model(trainOrTest, x, y, optim, l)
	parameterInfos = gather_parameter_infos(model)
	if(len(parameterInfos) == 0):
		return model(trainOrTest, x, y, optim, l)
	workerPerturbations = []
	fitnessValues = []
	rankValue = max(evolutionaryRank, 1)
	rankScale = 1.0 / math.sqrt(rankValue)
	signGroups = generate_antithetic_sign_groups(evolutionaryPopulationSize)
	signGroupSizes = [len(group) for group in signGroups]
	memberIndex = 0
	for signGroup in signGroups:
		basePerturbations = []
		for parameterInfo in parameterInfos:
			basePerturbations.append(sample_low_rank_factors(parameterInfo, rankValue))
		for sign in signGroup:
			currentSigma = get_sigma_for_member(memberIndex)
			memberIndex += 1
			scaledPerturbations = []
			for parameterInfo, perturbation in zip(parameterInfos, basePerturbations):
				apply_parameter_perturbation(parameterInfo, perturbation, currentSigma * sign, rankScale)
				scaledPerturbations.append(scale_perturbation(perturbation, sign))
			loss, _ = model(trainOrTest, x, y, optim, l)
			fitnessValues.append(-loss.item())
			for parameterInfo, perturbation in zip(parameterInfos, basePerturbations):
				apply_parameter_perturbation(parameterInfo, perturbation, -currentSigma * sign, rankScale)
			workerPerturbations.append(scaledPerturbations)
	fitnessWeights = calculate_fitness_weights(fitnessValues, parameterInfos[0]["tensor"].device, parameterInfos[0]["tensor"].dtype, signGroupSizes=signGroupSizes)
	apply_population_updates(parameterInfos, workerPerturbations, fitnessWeights, rankScale)
	return model(trainOrTest, x, y, optim, l)

def gather_parameter_infos(model):
	parameterInfos = []
	for parameter in model.parameters():
		if(not parameter.requires_grad):
			continue
		if(parameter.numel() == 0):
			continue
		rows, cols = determine_matrix_shape(parameter)
		info = {
			"tensor": parameter,
			"shape": parameter.shape,
			"rows": rows,
			"cols": cols
		}
		parameterInfos.append(info)
	return parameterInfos

def determine_matrix_shape(parameterTensor):
	if(parameterTensor.dim() == 0):
		return 1, 1
	if(parameterTensor.dim() == 1):
		return parameterTensor.shape[0], 1
	rows = parameterTensor.shape[0]
	cols = int(parameterTensor.numel() / rows)
	return rows, cols

def sample_low_rank_factors(parameterInfo, rankValue):
	rows = parameterInfo["rows"]
	cols = parameterInfo["cols"]
	tensor = parameterInfo["tensor"]
	A = pt.randn((rows, rankValue), device=tensor.device, dtype=tensor.dtype)
	B = pt.randn((cols, rankValue), device=tensor.device, dtype=tensor.dtype)
	return (A, B)

def apply_parameter_perturbation(parameterInfo, perturbation, sigmaScale, rankScale):
	A, B = perturbation
	tensor = parameterInfo["tensor"]
	updateMatrix = generate_low_rank_matrix(A, B, rankScale)
	tensor.add_(sigmaScale * updateMatrix.reshape(parameterInfo["shape"]))

def apply_population_updates(parameterInfos, workerPerturbations, fitnessWeights, rankScale):
	if(len(workerPerturbations) == 0):
		return
	if(fitnessWeights.numel() == 0):
		return
	updateScale = evolutionaryLearningRate / float(evolutionaryPopulationSize)
	for parameterIndex, parameterInfo in enumerate(parameterInfos):
		rows = parameterInfo["rows"]
		cols = parameterInfo["cols"]
		accumulatedUpdate = pt.zeros((rows, cols), device=parameterInfo["tensor"].device, dtype=parameterInfo["tensor"].dtype)
		for weight, workerPerturbation in zip(fitnessWeights, workerPerturbations):
			A, B = workerPerturbation[parameterIndex]
			accumulatedUpdate.add_(weight * generate_low_rank_matrix(A, B, rankScale))
		parameterInfo["tensor"].add_(updateScale * accumulatedUpdate.reshape(parameterInfo["shape"]))

def generate_low_rank_matrix(A, B, rankScale):
	return rankScale * pt.matmul(A, B.transpose(0, 1))

def scale_low_rank_factor(A, sign):
	return A * sign

def scale_perturbation(perturbation, sign):
	A, B = perturbation
	return (scale_low_rank_factor(A, sign), B)

def calculate_fitness_weights(fitnessValues, device, dtype, signGroupSizes=None):
	if(len(fitnessValues) == 0):
		return pt.zeros(0, device=device, dtype=dtype)
	fitnessTensor = pt.tensor(fitnessValues, device=device, dtype=dtype)
	shapingApplied = False
	if(evolutionaryFitnessShaping and evolutionaryAntitheticSampling and signGroupSizes is not None):
		shapedList = []
		currentIndex = 0
		for groupSize in signGroupSizes:
			if(groupSize == 2 and currentIndex+1 < fitnessTensor.numel()):
				s_pos = fitnessTensor[currentIndex]
				s_neg = fitnessTensor[currentIndex+1]
				score = pt.sign(s_pos - s_neg)
				shapedList.append(score)
				shapedList.append(-score)
			else:
				for _ in range(groupSize):
					shapedList.append(pt.tensor(0.0, device=device, dtype=dtype))
			currentIndex += groupSize
		if(len(shapedList) == fitnessTensor.numel()):
			fitnessTensor = pt.stack(shapedList)
			shapingApplied = True
	if(evolutionaryNormalizeFitness and fitnessTensor.numel() > 1 and not shapingApplied):
		fitnessTensor = fitnessTensor - fitnessTensor.mean()
		std = fitnessTensor.std(unbiased=False)
		if(std.item() < 1e-8):
			std = std + 1e-8
		fitnessTensor = fitnessTensor / std
	return fitnessTensor

def generate_antithetic_sign_groups(populationSize):
	if(populationSize <= 0):
		return []
	if(not evolutionaryAntitheticSampling):
		return [[1.0] for _ in range(populationSize)]
	groups = []
	remaining = populationSize
	while remaining > 0:
		if remaining >= 2:
			groups.append([1.0, -1.0])
			remaining -= 2
		else:
			groups.append([1.0])
			remaining -= 1
	return groups

def get_sigma_for_member(memberIndex):
	if(not evolutionaryPopulationSigmaScheduling or evolutionaryPopulationSize <= 1):
		return evolutionarySigma
	denominator = max(evolutionaryPopulationSize - 1, 1)
	fraction = memberIndex / denominator
	minScale = 0.25	#linearly decay sigma to 25% of original value across the population
	scale = (1.0 - fraction) * (1.0 - minScale) + minScale
	return evolutionarySigma * scale
