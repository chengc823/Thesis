import numpy as np
from typing import Protocol
from functools import partial
from scipy.stats import norm
from naslib.config import ACQType
from naslib.optimizers.bananas.calibrator import Distribution




def idependent_thompson_sampling(distribution: Distribution) -> float:
    return distribution.rvs(size=1)[0]


def upper_confidence_bound(distribution: Distribution, explore_factor: float = 0.5) -> float:
    return distribution.cdf(quantile=explore_factor)


def expected_improvement(distribution: Distribution, threshold: float) -> float:
    ...


def probability_of_improvement(distribution: Distribution, threhold: float) -> float:
    return 1 - distribution.ppf(quantile=threhold)


def exploit_only(distribution: Distribution) -> float:
    return distribution.mean()

    # if acq_fn_type == "its":
    #     # Independent Thompson sampling (ITS) acquisition function

    #     def its(arch_encoding, info=None):
    #         predictions = predictor.query([arch_encoding], info)
    #         predictions = np.squeeze(predictions)
    #         mean = np.mean(predictions)
    #         std = np.std(predictions)
    #         sample = np.random.normal(mean, std)
    #         return sample

    #     return its

    # elif acq_fn_type == "ucb":
    #     # Upper confidence bound (UCB) acquisition function

    #     def ucb(arch_encoding, info=None, explore_factor=0.5):

    #         predictions = predictor.query([arch_encoding], info)
    #         mean = np.mean(predictions)
    #         std = np.std(predictions)
    #         return mean + explore_factor * std

    #     return ucb

    # elif acq_fn_type == "ei":
    #     # Expected improvement (EI) acquisition function

    #     def ei(arch_encoding, info=None, ei_calibration_factor=5.0):
    #         predictions = predictor.query([arch_encoding], info)
    #         mean = np.mean(predictions)
    #         std = np.std(predictions)
    #         factored_std = std / ei_calibration_factor
    #         max_y = ytrain.max()
    #         gam = (mean - max_y) / factored_std
    #         ei_value = factored_std * (gam * norm.cdf(gam) + norm.pdf(gam))
    #         return ei_value

    #     return ei

    # elif acq_fn_type == "exploit_only":
    #     # Expected improvement (EI) acquisition function
        
    #     def exploit(arch_encoding, info=None):
    #         predictions = predictor.query([arch_encoding], info)
    #         return np.mean(predictions)

    #     return exploit

    # else:
    #     logger.info("{} is not a valid exploration type".format(acq_fn_type))
    #     raise NotImplementedError()
