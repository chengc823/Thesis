from __future__ import annotations

from typing import TypeAlias
from dataclasses import dataclass
from typing import TypeAlias
from numpy.typing import ArrayLike
import numpy as np
from naslib.search_spaces.core import Graph
from naslib.optimizers.bananas.distribution import Distribution


TrainCalDataSet: TypeAlias = tuple[list[Graph], list[Graph], list[float], list[float]]

@dataclass
class ConditionalEstimation:
    point_prediction: ArrayLike
    distribution: Distribution


# Calibration Score function
def calibration_metrics(obs_and_condest: list[tuple[float, ConditionalEstimation]], percentiles: list[float]) -> float:
    """A metric measures the precision of calibration."""
    def assess_single_quantile(obs_and_condest: list[tuple[float, ConditionalEstimation]], p):
        freq_p = 0
        for i, (obs, condest) in enumerate(obs_and_condest):
            if condest.distribution.cdf(obs) <= p:
                freq_p += 1
        score_p = freq_p / len(obs_and_condest)
        return (score_p - p)**2

    score = []
    for p_j in percentiles:
        p_j_score = assess_single_quantile(obs_and_condest=obs_and_condest, p=p_j)
        score.append(p_j_score)
    return np.sum(score)


# Conformity score functions
def conformity_scoring_normalise(value: float, mean: float, std: float) -> float:
    """Conformity scoring function based on normalising value."""
    assert std >= 0
    return (value - mean) / std

