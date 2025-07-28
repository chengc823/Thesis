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
    point_prediction: ArrayLike | dict[float, float]
    distribution: Distribution


# Calibration Score function
def rmsce_calibration(obs_and_condest: list[tuple[float, ConditionalEstimation]], percentiles: list[float]) -> float:
    """Return root mean squared calibration error.
    
    The calibration error consists of $(p_j - \tilde{p_j})^2$ where $p_j$ is a given quantile
    in [0, 1] and $\tilde{p_j}$ is the coverage of this quantile for the predictions, eg the percent of time this
    quantile was above the target.
    """
    cdfs = np.array([condest.distribution.cdf(obs) for obs, condest in obs_and_condest])
    score = []
    for p_j in percentiles:
        p_j_hat = sum(cdfs <= p_j) / len(cdfs)
        p_j_score = (p_j_hat - p_j) ** 2
        score.append(p_j_score)
    return np.sum(score)
