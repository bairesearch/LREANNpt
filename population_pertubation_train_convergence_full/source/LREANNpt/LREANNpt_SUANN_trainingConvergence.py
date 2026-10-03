"""Training- or validation-loss convergence for the SUANN experiment."""

import ANNpt_globalDefs as settings

populationPertubationOptimiseTrainingIterations = getattr(settings, "populationPertubationOptimiseTrainingIterations", False)
if(populationPertubationOptimiseTrainingIterations):
	import copy
	import math
	import torch as pt

	def _cpuCopy(value):
		if(isinstance(value, pt.Tensor)):
			return value.detach().cpu().clone()
		if(isinstance(value, dict)):
			return {key: _cpuCopy(item) for key, item in value.items()}
		if(isinstance(value, list)):
			return [_cpuCopy(item) for item in value]
		if(isinstance(value, tuple)):
			return tuple(_cpuCopy(item) for item in value)
		return copy.deepcopy(value)

	class PopulationPertubationTrainingConvergence:
		"""Select minimum full-split cross-entropy; exact ties keep the earlier model.

		The benchmark uses this same controller for its Adam comparison. A plateau
		is a practical stopping criterion, not a claim of globally optimal weights.
		Model and optimizer are restored together whenever learning rate is reduced.
		Validation is diagnostic in training-loss mode; test metrics are never supplied.
		"""

		def __init__(self, learningRate, minimumCompletePassIterations=0):
			self.trainSetLossOptimisation = settings.trainSetLossOptimisation
			if(type(self.trainSetLossOptimisation) is not bool):
				raise ValueError("trainSetLossOptimisation must be a boolean")
			self.selectionSplit = "train" if self.trainSetLossOptimisation else "validation"
			self.selection = "minimum full-" + ("training" if self.trainSetLossOptimisation else "validation") + " cross-entropy"
			prefix = "populationPertubation" + ("Training" if self.trainSetLossOptimisation else "Validation")
			self.policy = {
				"trainSetLossOptimisation": self.trainSetLossOptimisation,
				"selection_metric": self.selectionSplit + "_loss",
				"evaluate_every": settings.populationPertubationEvaluateEveryIterations,
				"minimum_updates": max(settings.populationPertubationMinimumTrainingIterations, minimumCompletePassIterations),
				"patience": getattr(settings, prefix + "Patience"),
				"min_delta": getattr(settings, prefix + "MinDelta"),
				"relative_min_delta": getattr(settings, prefix + "RelativeMinDelta"),
				"lr_factor": settings.populationPertubationTrainingLearningRateFactor,
				"lr_reductions": settings.populationPertubationTrainingLearningRateReductions,
				"max_numerical_recoveries": settings.populationPertubationTrainingMaxNumericalRecoveries,
			}
			if(self.trainSetLossOptimisation):
				self.policy["loss_goal"] = settings.populationPertubationTrainingLossGoal
			for name in ("evaluate_every", "minimum_updates", "patience"):
				if(type(self.policy[name]) is not int or self.policy[name] < 1):
					raise ValueError(name + " must be a positive integer")
			for name in ("lr_reductions", "max_numerical_recoveries"):
				if(type(self.policy[name]) is not int or self.policy[name] < 0):
					raise ValueError(name + " must be a nonnegative integer")
			for name in ("min_delta", "relative_min_delta") + (("loss_goal",) if self.trainSetLossOptimisation else ()):
				if(not math.isfinite(self.policy[name]) or self.policy[name] < 0):
					raise ValueError(name + " must be finite and nonnegative")
			if(not 0 < self.policy["lr_factor"] < 1 or not math.isfinite(learningRate) or learningRate <= 0):
				raise ValueError("Learning rate must be positive and its reduction factor must be between zero and one")
			self.learningRate = learningRate
			self.reductions = 0
			self.numericalRecoveries = 0
			self.best = None
			self.significantLoss = math.inf
			self.lastSignificantIteration = 0
			self.lastObservation = -1
			self.stopReason = None
			self.events = []

		def shouldEvaluate(self, iteration):
			return iteration % self.policy["evaluate_every"] == 0

		def restoreBest(self, model, optimizer=None):
			if(self.best is None):
				raise RuntimeError("No finite " + self.selectionSplit + "-selected checkpoint is available")
			model.load_state_dict(self.best["model"])
			if(optimizer is not None):
				optimizer.load_state_dict(self.best["optimizer"])
				for group in optimizer.param_groups:
					group["lr"] = self.learningRate

		def _reduceLearningRate(self, iteration, model, optimizer, reason, message=None):
			oldRate = self.learningRate
			self.learningRate *= self.policy["lr_factor"]
			self.reductions += 1
			self.restoreBest(model, optimizer)
			self.significantLoss = self.best[self.selectionSplit]["loss"]
			self.lastSignificantIteration = iteration
			self.events.append({"step": iteration, "event": reason, "old_lr": oldRate,
				"lr": self.learningRate, "restored_step": self.best["step"], "message": message})

		def observe(self, iteration, trainMetrics, validationMetrics, model, optimizer=None):
			if(self.stopReason is not None or iteration <= self.lastObservation):
				raise ValueError("Training observations must advance and must precede stopping")
			if(not self.trainSetLossOptimisation and validationMetrics is None):
				raise ValueError("Independent validation metrics are required for convergence")
			checkedMetrics = [("training", trainMetrics)]
			if(not self.trainSetLossOptimisation):
				checkedMetrics.append(("validation", validationMetrics))
			for split, metrics in checkedMetrics:
				if(not math.isfinite(float(metrics["loss"])) or not math.isfinite(float(metrics["accuracy"])) or not 0 <= float(metrics["accuracy"]) <= 1):
					raise FloatingPointError("Non-finite or invalid full-" + split + " metrics")
			loss = float((trainMetrics if self.trainSetLossOptimisation else validationMetrics)["loss"])
			if(any(not pt.isfinite(value).all().item() for value in model.state_dict().values() if value.is_floating_point())):
				raise FloatingPointError("Non-finite model state at training evaluation")
			self.lastObservation = iteration
			if(self.best is None or loss < self.best[self.selectionSplit]["loss"]):
				self.best = {"step": iteration, "train": copy.deepcopy(trainMetrics),
					"validation": copy.deepcopy(validationMetrics),
					"model": _cpuCopy(model.state_dict()),
					"optimizer": _cpuCopy(optimizer.state_dict()) if optimizer is not None else None}
			threshold = max(self.policy["min_delta"], self.policy["relative_min_delta"] * self.significantLoss)
			if(not math.isfinite(self.significantLoss) or loss < self.significantLoss - threshold):
				self.significantLoss = loss
				self.lastSignificantIteration = iteration
			if(iteration >= self.policy["minimum_updates"]):
				if(self.trainSetLossOptimisation and float(trainMetrics["accuracy"]) == 1.0 and loss <= self.policy["loss_goal"]):
					self.stopReason = "perfect_train_fit"
				elif(iteration - self.lastSignificantIteration >= self.policy["patience"]):
					label = "training" if self.trainSetLossOptimisation else "validation"
					if(self.reductions >= self.policy["lr_reductions"]):
						self.stopReason = label + "_loss_plateau_after_lr_reductions"
					else:
						self._reduceLearningRate(iteration, model, optimizer, label + "_plateau_reduce_lr_restore_best")
			return self.stopReason

		def recoverNonfinite(self, iteration, model, optimizer=None, message="Non-finite training update"):
			if(self.best is None or self.numericalRecoveries >= self.policy["max_numerical_recoveries"]):
				raise FloatingPointError("Numerical recovery exhausted; this run has not converged: " + message)
			self.numericalRecoveries += 1
			self._reduceLearningRate(iteration, model, optimizer, "numerical_recovery_reduce_lr_restore_best", message)

		def state_dict(self):
			return _cpuCopy(vars(self))

		def load_state_dict(self, state):
			if(state["policy"] != self.policy):
				raise ValueError("Cannot resume with a different convergence policy")
			self.__dict__.update(_cpuCopy(state))

	def _populationDataset(dataset):
		if(settings.useTabularDataset):
			import ANNpt_data
			return ANNpt_data.DataloaderDatasetTabular(dataset)
		if(settings.useImageDataset):
			return dataset
		raise ValueError("Population convergence supports tabular and image datasets")

	@pt.no_grad()
	def evaluatePopulationPertubationDataset(dataset, model):
		#No repeat sampler, random transforms introduced here, or dropped final batch.
		loader = pt.utils.data.DataLoader(_populationDataset(dataset), batch_size=settings.batchSize,
			shuffle=False, drop_last=False, generator=pt.Generator().manual_seed(0))
		if(len(loader.dataset) == 0):
			raise ValueError("Cannot evaluate an empty dataset")
		training = model.training
		model.eval()
		lossSum = 0.0
		correct = 0
		rows = 0
		try:
			for x, y in loader:
				x, y = x.to(settings.device), y.long().to(settings.device)
				loss, _ = model(False, x, y, None, None)
				lossSum += float(loss) * len(y)
				correct += int((model.Ztrace[-1].argmax(1) == y).sum())
				rows += len(y)
		finally:
			model.train(training)
		return {"loss": lossSum / rows, "accuracy": correct / rows, "rows": rows}

	def trainPopulationPertubationUntilConverged(dataset, validationDataset, model, algorithm):
		if(not settings.trainSetLossOptimisation and (validationDataset is None or validationDataset is dataset)):
			raise ValueError("A separate validation dataset is required; never pass the training or test dataset")
		trainingData = _populationDataset(dataset)
		if(len(trainingData) == 0):
			raise ValueError("Cannot train on an empty dataset")
		loader = pt.utils.data.DataLoader(trainingData, batch_size=settings.batchSize,
			shuffle=True, drop_last=False, generator=pt.Generator().manual_seed(pt.initial_seed()))
		controller = PopulationPertubationTrainingConvergence(algorithm.populationPertubationLearningRate, len(loader))
		iteration = 0
		controller.observe(0, evaluatePopulationPertubationDataset(dataset, model),
			None if controller.trainSetLossOptimisation else evaluatePopulationPertubationDataset(validationDataset, model), model)
		initialRate = algorithm.populationPertubationLearningRate
		try:
			while(controller.stopReason is None):
				for x, y in loader:
					model.train()
					iteration += 1
					x, y = x.to(settings.device), y.long().to(settings.device)
					try:
						loss, _ = algorithm.trainOrTestModel(model, True, x, y, None, None)
						if(not pt.isfinite(loss).item()):
							raise FloatingPointError("Non-finite updated-model loss")
						if(controller.shouldEvaluate(iteration)):
							metrics = evaluatePopulationPertubationDataset(dataset, model)
							#Avoid extra evaluations/RNG consumption in the original training-only path.
							validation = None if controller.trainSetLossOptimisation else evaluatePopulationPertubationDataset(validationDataset, model)
							controller.observe(iteration, metrics, validation, model)
							print("population training", iteration, "train", metrics, "validation", validation, "learningRate", controller.learningRate, flush=True)
					except FloatingPointError as error:
						controller.recoverNonfinite(iteration, model, message=str(error))
						print("population numerical recovery", controller.events[-1], flush=True)
					algorithm.populationPertubationLearningRate = controller.learningRate
					if(controller.stopReason is not None):
						break
			controller.restoreBest(model)
			result = {"iterations": iteration, "selected_iteration": controller.best["step"],
				"stop_reason": controller.stopReason, "train": controller.best["train"],
				"validation": controller.best["validation"],
				"learning_rate_events": controller.events, "policy": controller.policy,
				"selection": controller.selection, "global_optimum_proven": False}
			model.populationPertubationTrainingResult = result
			return result
		finally:
			algorithm.populationPertubationLearningRate = initialRate
