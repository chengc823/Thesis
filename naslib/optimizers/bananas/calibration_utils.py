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
    cdfs = np.array([condest.distribution.cdf(obs) for obs, condest in obs_and_condest])
    score = []
    for p_j in percentiles:
        p_j_hat = sum(cdfs <= p_j) / len(cdfs)
        p_j_score = (p_j_hat - p_j) ** 2
        score.append(p_j_score)
    return np.sum(score)


# Conformity score functions
def conformity_scoring_normalise(value: float, mean: float, std: float) -> float:
    """Conformity scoring function based on normalising value."""
    assert std >= 0
    return (value - mean) / std

