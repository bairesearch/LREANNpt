# LREANNpt

### Author

Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

### Description

Learning Rule Experiment artificial neural network (LREANN) for PyTorch - experimental 

* SUANN - stochastic update artificial neural network:
  * Stochastic search - perturb individual matrix weights and measure loss
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

### References

* https://github.com/bairesearch/LREANNtf (SUANN)
* https://github.com/bairesearch/EISANIpt (useStochasticUpdates)
* Sarkar, B., Fellows, M., Duque, J. A., Letcher, A., Villares, A. L., Sims, A., ... & Foerster, J. N. (2025). Evolution Strategies at the Hyperscale. arXiv preprint arXiv:2511.16652.
