from abc import ABC, abstractmethod
from typing import Type, Callable, Literal
import scipy.stats as stats
import math
import numpy as np
from sklearn.model_selection import train_test_split
from naslib.search_spaces.core import Graph
from naslib.predictors.base import Predictor
from naslib.predictors.ensemble import Ensemble
from naslib.config import CalibratorType
from naslib.optimizers.bananas.distribution import Distribution, PointwiseInterpolatedDist


def normalise(value: float, mean: float, std: float) -> float:
    assert std >= 0
    return (value - mean) / std


class BaseCalibrator(ABC):
    """Bluprint of Conformal Prediction based calibrator.

    predictor: point estimator to be fitted and calibrated.
    train_cal_split: approach to split the dataset into a training set and a calibration set.
    seed: random seed for splitting data set.
    """

    def __init__(self, predictor: Predictor, train_cal_split: float | None ,seed: int = 42):
        self.predictor = predictor
        self.train_cal_split = train_cal_split
        self.conformity_func = normalise   # TODO: make the conformity score function a constant for now
        self.seed = seed
        self._is_calibrated = False

    @abstractmethod
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        raise NotImplementedError
    
    @abstractmethod
    def get_distribution(self, data: Graph, percentiles: list[float] | None = [0.05, 0.1, 0.5, 0.9, 0.95]) -> Distribution:
        """Get the distribution confitional on the given data point.
        
        Note: percentiles is only required if the distribution is discrete.
        """
        raise NotImplementedError 


class Gaussian(BaseCalibrator):
    
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        X_train, y_train = data
        self.predictor.fit(X_train, y_train)

    def get_distribution(self, data: Graph, percentiles=None):
        predictions = np.squeeze(self.predictor.query([data]))
        mean = np.mean(predictions)
        std = np.std(predictions)
        return stats.norm(loc=mean, scale=std)
    

class SplitCPCalibrator(BaseCalibrator):

    def _split(self, X: list[Graph], y: list[float]) -> tuple[list[Graph], list[Graph], list[float], list[float]]:
        """Split the dataset into a train set and a calibration set
        
        Returns: a tuple representing (X_train, X_cal, y_train, y_cal).
        """
        # get trainaing set size and calibration set size
        cal_size = int(len(X) * self.train_cal_split)
        # randomly sample calibration set
        obs_indices = np.arange(0, len(X))
        train_indices, cal_indices = train_test_split(obs_indices, test_size=cal_size, random_state=self.seed)
        print(f"Train set size: {len(train_indices)}; Calibration set size: {len(cal_indices)}")

        X_train, X_cal, y_train, y_cal = [], [], [], []
        for idx in obs_indices:
            X_i, y_i = X[idx], y[idx]
            if idx in train_indices:
                X_train.append(X_i)
                y_train.append(y_i)
            elif idx in cal_indices:
                X_cal.append(X_i)
                y_cal.append(y_i)
        
        return X_train, X_cal, y_train, y_cal

    def calibrate(self, data: tuple[list[Graph], list[float]]):
        X, y = data
        # split the data into train and validate
        X_train, X_cal, y_train, y_cal = self._split(X=X, y=y)
        # fit the predictor
        self.predictor.fit(X_train, y_train)
        # calbrate 
        self.conformity_scores = []
        for X_i, y_i in zip(X_cal, y_cal):
            if isinstance(self.predictor, Ensemble):
                preds_i = np.squeeze(self.predictor.query([X_i]))
                mean_i = np.mean(preds_i)
                std_i = np.std(preds_i)
            else:
                raise NotImplementedError
            self.conformity_scores.append(self.conformity_func(value=y_i, mean=mean_i, std=std_i))

        self.num_seen_obs = len(X)
        self._is_calibrated = True

    def get_distribution(self, data: Graph, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> PointwiseInterpolatedDist:
        assert self._is_calibrated

        if isinstance(self.predictor, Ensemble):
            preds = np.squeeze(self.predictor.query([data]))
            mean = np.mean(preds)
            std = np.std(preds)
        else:
            raise NotImplementedError

        quantiles = []
        for p in percentiles:
            n_cal = len(self.conformity_scores)
            adj_p = min(math.ceil((n_cal + 1) * p) / n_cal, 1.0)
            quantile = np.quantile(self.conformity_scores, adj_p) * std + mean
            quantiles.append(quantile)
        return PointwiseInterpolatedDist(values=(percentiles, quantiles))



def get_calibrator_class(calibrator_type: CalibratorType,) -> Type[BaseCalibrator]:
    match calibrator_type:
        case CalibratorType.GAUSSIAN:
            return Gaussian
        case CalibratorType.CP_SPLIT:
            return SplitCPCalibrator
    