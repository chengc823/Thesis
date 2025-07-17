# Uncertainty Calibration for Neural Architecture Search
This repository provides the code for the thesis project in FFS/2025 at the University of Mannheim with the topic:

*Uncertainty Calibration with Online Conformal Prediction in Neural Architecture Search: An Evaluation under the BANANAS Framework*

This thesis proposes a new framework BANANAS-CP that incoporates a uncertainty calibration process into the high-performing [BANANAS](https://arxiv.org/abs/1910.11858) framework and has conducted experiments on the tabular benchmark dataset [NAS-Bench-201](https://arxiv.org/abs/2001.00326). Despite of several interesting findings, experiment results indicate that in general uncertainty calibration using conformal prediction does not improve the search efficiency.

### Evaluate BANANAS-CP
**Setup**: clone the repository and create an evironment using the ``environment.yml``:
```
git clone https://github.com/chengc823/Thesis.git
conda env create -f environment.yml
```
**Data**: download the benchmark data files using the associated URLs provided by [NASLib](https://github.com/automl/NASLib/tree/Develop) and place the data files in `./naslib/data`.
- Cifar10: [naslib/data/nb201_cifar10_full_training.pickle](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view)
- Cifar100: [naslib/data/nb201_cifar100_full_training.pickle](https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view)
- ImageNet16-120: [naslib/data/nb201_ImageNet16_full_training.pickle](https://drive.google.com/file/d/1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH/view)
 
**Run Experiments**: setup the configurations with ``./experiments/config.yml`` (descriptions of parameters can be found in ``./naslib/config.py``). After setting up the config file, checkout to ``./experiments/demo.py`` and execute.


