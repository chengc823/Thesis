from abc import ABC, abstractmethod
from typing import Type, Callable, Literal
from typing_extensions import Self
from numpy.typing import ArrayLike
import scipy.stats as stats
import numpy as np
from naslib.search_spaces.core import Graph
from naslib.predictors.base import Predictor
from naslib.config import CalibratorType
from naslib.optimizers.bananas.distribution import Distribution


def conformity_scoring(value: float, mean: float, std: float, mode: Literal["normalise"] = "normalise") -> float:
    assert std >= 0
    
    if mode == "normalise": 
        return abs(value - mean) / std


class BaseCalibrator(ABC):
    """Bluprint of Conformal Prediction based calibrator.

    predictor: The point estimator to be fitted and calibrated.
    train_cal_split: Approach to split the dataset into a training set and a calibration set.
    conformity_score: A function that assigns conformity score to each point in the calibration set.
    percentiles: A list of quantiles at which the conformity scores are computed.
    """

    def __init__(self, 
        predictor: Predictor,
        train_cal_split: float | None ,
        conformity_score: Callable = conformity_scoring,
        percentiles: list[float] = [0.05, 0.1, 0.5, 0.9, 0.95],
    ):
        self.predictor = predictor
        self.train_cal_split = train_cal_split,
        self.comformity_score = conformity_score
        self.percentiles = percentiles

    @abstractmethod
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        raise NotImplementedError
    
    @abstractmethod
    def get_distribution(self, data: Graph) -> Distribution:
        raise NotImplementedError 



class Gaussian(BaseCalibrator):
    
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        xtrain, ytrain = data
        self.predictor.fit(xtrain, ytrain)

    def get_distribution(self, data: Graph):
        predictions = self.predictor.query([data])
        predictions = np.squeeze(predictions)
        mean = np.mean(predictions)
        std = np.std(predictions)
        return stats.norm(loc=mean, scale=std)




def get_calibrator_class(calibrator_type: CalibratorType,) -> Type[BaseCalibrator]:
    if calibrator_type == CalibratorType.GAUSSIAN:
        return Gaussian