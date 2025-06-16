from abc import ABC, abstractmethod
from typing import Type
import torch.nn as nn
from numpy.typing import ArrayLike
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from naslib.search_spaces.core import Graph
from naslib.predictors.base import Predictor
from naslib.predictors.ensemble import Ensemble
from naslib.predictors.quantile_regressor import QuantileRegressor
from naslib.config import CalibratorType, PredictorType
from naslib.optimizers.bananas.distribution import GaussianDist, PointwiseInterpolatedDist
from naslib.optimizers.bananas.calibration_utils import conformity_scoring_normalise, ConditionalEstimation, TrainCalDataSet


class BaseCalibrator(ABC):
    """Bluprint of Conformal Prediction based calibrator.

    predictor: point estimator to be fitted and calibrated.
    train_cal_split: approach to split the dataset into a training set and a calibration set.
    seed: random seed for splitting data set.
    """

    def __init__(self, predictor: Predictor, train_cal_split: float | None ,seed: int = 42):
        self.predictor = predictor
        self.train_cal_split = train_cal_split
        self.seed = seed
        self._is_calibrated = False

    @abstractmethod
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        raise NotImplementedError
    
    @abstractmethod
    def get_conditional_estimation(self, data: Graph, percentiles: list[float] | None = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        """Get the distribution conditional on the given data point.
        
        Note: percentiles is only required if the distribution is discrete.
        """
        raise NotImplementedError 


class Gaussian(BaseCalibrator):
    
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        X_train, y_train = data
        self.predictor.fit(X_train, y_train, loss=nn.L1Loss())

    def get_conditional_estimation(self, data: Graph, percentiles=None) -> ConditionalEstimation:
        predictions = np.squeeze(self.predictor.query([data]))
        mean = np.mean(predictions)
        std = np.std(predictions)
        return ConditionalEstimation(point_prediction=predictions, distribution=GaussianDist(loc=mean, scale=std))
    

def _get_train_cal_dataset(X: list[Graph], y: list[float], train_indices: ArrayLike, cal_indices: ArrayLike) -> TrainCalDataSet:
    X_train, X_cal, y_train, y_cal = [], [], [], []
    for idx in range(len(X)):
        X_i, y_i = X[idx], y[idx]
        if idx in train_indices:
            X_train.append(X_i)
            y_train.append(y_i)
        elif idx in cal_indices:
            X_cal.append(X_i)
            y_cal.append(y_i)
    
    return X_train, X_cal, y_train, y_cal

class EnsembleCalibrationMixin:
    """
        - Conformity scoring function: (y - y_hat) / sd_hat.
    """
    def get_conformity_score(self, predictor: Ensemble, X_i, y_i) -> float:
        preds_i = np.squeeze(predictor.query([X_i]))
        mean_i = np.mean(preds_i)
        std_i = np.std(preds_i)
        return conformity_scoring_normalise(value=y_i, mean=mean_i, std=std_i)

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        preds = np.squeeze(self.predictor.query([data]))
        mean = np.mean(preds)
        std = np.std(preds)
        
        quantiles = []
        for p in percentiles:
            n_cal = len(self.conformity_scores)
            adj_p = min((n_cal + 1) * p / n_cal, 1.0)  # adjusted percentil for finite sample
            quantile = np.quantile(self.conformity_scores, adj_p) * std + mean
            quantiles.append(quantile)

        return ConditionalEstimation(point_prediction=preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))
    

class SplitCPMixin:
    
    def _split(self, X: list[Graph], y: list[float]) -> TrainCalDataSet:
        """Split the dataset into a train set and a calibration set"""

        # get trainaing set size and calibration set size
        cal_size = int(len(X) * self.train_cal_split)
        # randomly sample calibration set
        obs_indices = np.arange(0, len(X))
        train_indices, cal_indices = train_test_split(obs_indices, test_size=cal_size, random_state=self.seed)
        print(f"Train set size={len(train_indices)}; Calibration set size={len(cal_indices)}")
        
        return _get_train_cal_dataset(X=X, y=y, train_indices=train_indices, cal_indices=cal_indices)


class EnsembleSplitCPCalibrator(SplitCPMixin, EnsembleCalibrationMixin, BaseCalibrator):
    """Uncertainty calibrator based on ensemble predictor and split Conformal Prediction."""
    
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        """Get conformity scores using the calibration dataset."""
        X, y = data
        self.num_seen_obs = len(X)
        # split the data into train and validate
        X_train, X_cal, y_train, y_cal = self._split(X=X, y=y)
        # fit the predictor
        self.predictor.fit(X_train, y_train, loss=nn.L1Loss())
        # calbrate 
        self.conformity_scores = [self.get_conformity_score(predictor=self.predictor, X_i=X_i, y_i=y_i) for X_i, y_i in zip(X_cal, y_cal)]
        self._is_calibrated = True


class CrossValCPMixin:

    def _cross_val_split(self, X: list[Graph], y: list[float]) -> list[TrainCalDataSet]:
        obs_indices = np.arange(0, len(X))
        kfolds = KFold(n_splits=self.train_cal_split).split(obs_indices)
       
        data_folds = []
        for i, (train_indices, cal_indices) in enumerate(kfolds):
            print(f"Running fold {i}: train set size={len(train_indices)}; calibration set size={len(cal_indices)}")
            train_cal_dataset = _get_train_cal_dataset(X=X, y=y, train_indices=train_indices, cal_indices=cal_indices)
            data_folds.append(train_cal_dataset)
        return data_folds


class EnsembleCrossValCPCalibrator(CrossValCPMixin, EnsembleCalibrationMixin, BaseCalibrator):
    """Uncertainty calibrator based on ensemble predictor and cross-validation based Conformal Prediction."""
                 
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        X, y = data
        self.num_seen_obs = len(X)
        cv_splits = self._cross_val_split(X=X, y=y)

        self.conformity_scores = []
        for X_train, X_cal, y_train, y_cal in cv_splits:
            self.predictor.fit(X_train, y_train, loss=nn.L1Loss())
            for X_i, y_i in zip(X_cal, y_cal):
                self.conformity_scores.append(self.get_conformity_score(predictor=self.predictor, X_i=X_i, y_i=y_i))

        assert len(self.conformity_scores) == len(X)
        self._is_calibrated = True



def get_calibrator_class(predictor_type: PredictorType, calibrator_type: CalibratorType) -> Type[BaseCalibrator]:
    calibrator_map = {
        (PredictorType.ENSEMBLE_MLP, CalibratorType.GAUSSIAN): Gaussian,
        (PredictorType.ENSEMBLE_MLP, CalibratorType.CP_SPLIT): EnsembleSplitCPCalibrator,
        (PredictorType.ENSEMBLE_MLP, CalibratorType.CP_CROSSVAL): EnsembleCrossValCPCalibrator
    }
    return calibrator_map[(predictor_type, calibrator_type)]