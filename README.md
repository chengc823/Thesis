# Uncertainty Estimation for Neural Architecture Search
This repository provides the code for the thesis project in FFS/2025 at the University of Mannheim with the topic:

*Uncertainty Estimation with Online Conformal Prediction in Neural Architecture Search: An Evaluation under the BANANAS Framework*

### Information
This thesis proposes a new framework BANANAS-CP, which is an extension of the high-performing NAS framework, [BANANAS](https://arxiv.org/abs/1910.11858), with an additional uncertainty calibration process based on conformal prediction (CP). Specically, BANANAS-CP supports six various setups:

- Pure BANANAS without calibration
- Split CP with ensemble predictor
- Cross-validation CP with ensemble predictor
- Bootstrapping CP with ensemble predictor
- Split CP with quantile regressor
- Cross-validation CP with quantile regressor 

### Evaluation
The current version of BANANAS-CP supports running experiments on the tabular benchmark dataset [NAS-Bench-201](https://arxiv.org/abs/2001.00326).

**Setup**: clone the repository and create an evironment using the ``environment.yml`` file:
```
git clone https://github.com/chengc823/Thesis.git
conda env create -f environment.yml
```

**Data**: download NAS-Bench-201 data files using the associated URLs provided by [NASLib](https://github.com/automl/NASLib/tree/Develop) and place the data files in `./naslib/data`.
- CIFAR10: [naslib/data/nb201_cifar10_full_training.pickle](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view)
- CIFAR100: [naslib/data/nb201_cifar100_full_training.pickle](https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view)
- ImageNet16-120: [naslib/data/nb201_ImageNet16_full_training.pickle](https://drive.google.com/file/d/1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH/view)
 
**Configure**: Checkout to ``./experiments/config.yml`` and set up relevant configurations, like calibration algorithm, prediction algorithm, datset, etc. This ``config.yml`` file is deserialized and validated by a pydantic-based model. Descriptions of parameters can be found in ``./naslib/config.py``.

**Run**: Checkout to ``./experiments/demo.py``, set up simulation parameters, and execute.


