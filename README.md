# LREANNpt

### Author

Richard Bruce Baxter - Copyright (c) 2023-2026 BAI Research (bairesearch.com.au)

### Description

Learning Rule Experiment artificial neural network (LREANN) for PyTorch - experimental 

* SUANN - stochastic update artificial neural network:
  * Individual perturbation (stochastic search) - perturb individual matrix weights and measure loss
  * Population perturbation - jointly perturb all trainable parameters and average reward-weighted Gaussian directions
  * Evolutionary search/strategies (ES) - Evolution Guided General Optimization via Low-rank Learning (EGGROLL)

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics
pip install torchvision
pip install torchsummary
pip install networkx
pip install matplotlib
pip install transformers
pip install h5py
pip install spacy
python -m spacy download en_core_web_sm
```

### Execution
```
source activate pytorchsenv
python ANNpt_main.py
```

### SUANN population perturbation

In `LREANNpt/LREANNpt_SUANN_globalDefs.py`, enable `useStochasticUpdates` and set `usePopulationPertubation = True`. New implementation code and settings are guarded by `if(usePopulationPertubation)`.

* `populationPertubationPopulationSize = 64`: number of perturbed models evaluated per minibatch (for example 64, 256, 1024 or 4096; must be an integer >= 2).
* `populationPertubationSigma = 0.01`: Gaussian noise standard deviation per parameter.
* `populationPertubationLearningRate = 0.01`: step size for the averaged update.

Each member starts from the same original model and simultaneously perturbs **every scalar in every trainable parameter tensor**, across all layers, including biases and BatchNorm affine parameters. Frozen parameters and buffers are excluded. Directions and their scalar entries are independent standard Gaussian samples, with no low-rank factorisation, coordinate selection, antithetic pairing, or fitness standardisation.

For member `i`, evaluate `reward_i = -loss(theta + sigma * epsilon_i)` on the same minibatch. Then apply one update:

```text
theta += learningRate / (populationSize * sigma)
         * sum((reward_i - mean(rewards)) * epsilon_i)
```

Thus a lower-than-average loss gives that member's noise a positive weight. The mean reward is a baseline; parameter updates average the reward-weighted noise, rather than just averaging losses. The Gaussian estimator and seed replay follow [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864), with population-mean centring as specified in `dev/populationPertubationUpgradeInfo.txt`.

Candidates are evaluated sequentially under `torch.no_grad()`, replaying seeds to reconstruct directions. Storage scales with model size plus population size, rather than their product. All candidates share the same initial buffers and PyTorch random state, so dropout noise and BatchNorm updates do not favour individual members. Parameters and buffers are restored exactly after candidate evaluation, including on failure. Only the final updated-model pass accumulates SUANN accuracy metrics and running statistics; its loss and accuracy are returned. An iteration costs `populationSize + 1` forward passes.

Larger populations can reduce estimator noise but increase computation. The supplied graph plots validation loss against steps, not runtime, and supplies no dataset, model or full training recipe. These defaults implement the described estimator; matching or beating backpropagation on a particular task requires benchmarking and tuning sigma and learning rate.

With `populationPertubationOptimiseTrainingIterations=True`, `trainSetLossOptimisation` controls checkpoint selection and stopping. `False` (the default) selects minimum validation cross-entropy and uses validation-loss plateaus to trigger learning-rate reductions and eventually stop training; validation accuracy does not influence these decisions. `True` restores the original training-loss policy, including stopping at 100% training accuracy and training cross-entropy ≤ 0.01 after the minimum training period. Equal selected-split losses retain the earlier checkpoint. In validation-loss mode, the production loader holds out validation data from training before preprocessing when no validation split is provided; training-loss mode skips this added holdout. Test data is used only for final evaluation.

The full-data benchmark reads `trainSetLossOptimisation` from `protocol.json` and applies the selected loss-based controller to population perturbation and Adam, retaining its existing train/validation/test partitions in both modes. Its final report includes train, validation, and test accuracy for every dataset and method, plus elapsed run time as mean ± sample standard deviation over three seeds. Start a fresh run to adopt this protocol or change modes; existing checkpoints cannot be resumed under a different policy. See [benchmark instructions](population_pertubation_train_convergence_full/README.txt).

### References

* https://github.com/bairesearch/LREANNtf (SUANN)
* https://github.com/bairesearch/EISANIpt (useStochasticUpdates)
* Sarkar, B., Fellows, M., Duque, J. A., Letcher, A., Villares, A. L., Sims, A., ... & Foerster, J. N. (2025). Evolution Strategies at the Hyperscale. arXiv preprint arXiv:2511.16652.
