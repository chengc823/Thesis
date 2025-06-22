from abc import ABC, abstractmethod
import copy
from typing import Type
import torch.nn as nn
from numpy.typing import ArrayLike
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from naslib.search_spaces.core import Graph
from naslib.predictors.base import Predictor
from naslib.predictors.mlp import MLPPredictor
from naslib.predictors.ensemble import Ensemble
from naslib.predictors.quantile_regressor import QuantileRegressor
from naslib.config import CalibratorType, PredictorType
from naslib.optimizers.bananas.distribution import GaussianDist, PointwiseInterpolatedDist
from naslib.optimizers.bananas.calibration_utils import ConditionalEstimation, TrainCalDataSet


class BaseCalibrator(ABC):
    """Blueprint of Conformal Prediction based calibrator.

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
    @staticmethod
    def conformity_score_fn(value: float, mean: float, std: float) -> float:
        """Conformity scoring function based on normalising value."""
        assert std >= 0
        return abs(value - mean) / std

    def get_conformity_score(self, predictor: Ensemble, X_i, y_i) -> float:
        preds_i = np.squeeze(predictor.query([X_i]))
        mean_i = np.mean(preds_i)
        std_i = np.std(preds_i)
        return self.conformity_score_fn(value=y_i, mean=mean_i, std=std_i)
    

class QuantileCalibrationMixin:
    """
        - Conformity scoring function: max(Q_alpha(x)_hat - y, y - Q_(1-alpha)(x)).
    """
    @staticmethod
    def conformity_score_fn(value: float, quantile_pred: float, alpha: float):
    # def conformity_score_fn(value: float, quantile_preds: dict[float, float], alpha: float):
        """Conformity scoring function for CQR."""
        def _get_sign(alpha: float):
            if alpha == 0.5:
                return 0.0
            elif alpha < 0.5:
                return 1.0
            elif alpha > 0.5:
                return -1.0
            
        sign = _get_sign(alpha=alpha)
        return sign * (quantile_pred - value)

    def get_conformity_score(self, predictor: QuantileRegressor, X_i, y_i) -> dict[float, float]:
        quantile_preds = predictor.query([X_i])
        conformity_scores = {}
        for alpha, pred in quantile_preds.items():
            conformity_scores[alpha] = self.conformity_score_fn(value=y_i, quantile_pred=pred, alpha=alpha)
        return conformity_scores


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
        # split the data into train and validate
        X_train, X_cal, y_train, y_cal = self._split(X=X, y=y)
        # fit the predictor
        self.predictor.fit(X_train, y_train, loss=nn.L1Loss())
        # calbrate 
        self.conformity_scores = [self.get_conformity_score(predictor=self.predictor, X_i=X_i, y_i=y_i) for X_i, y_i in zip(X_cal, y_cal)]
        self._is_calibrated = True

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        preds = np.squeeze(self.predictor.query([data]))
        mean = np.mean(preds)
        std = np.std(preds)
    
        quantiles = []
        for p in percentiles:
            n_cal = len(self.conformity_scores)
            if p < 0.5:
                target_p = min((1 - 2 * p) * (1 + 1 / n_cal), 1) # adjusted percentil for finite sample
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean - correction
            elif p > 0.5:
                target_p = min((2 * p - 1) * (1 + 1 / n_cal), 1)
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean + correction
            else:
                quantile = mean
            quantiles.append(quantile)
        return ConditionalEstimation(point_prediction=preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))


class QuantileSplitCPCalibrator(SplitCPMixin, QuantileCalibrationMixin, BaseCalibrator):
    """Uncertainty calibrator based on ensemble predictor and split Conformal Prediction."""
    
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        """Get conformity scores using the calibration dataset."""
        X, y = data
        # split the data into train and validate
        X_train, X_cal, y_train, y_cal = self._split(X=X, y=y)
        # fit the predictor
        self.predictor.fit(X_train, y_train)
        # calbrate 
        self.conformity_scores = {alpha: [] for alpha in self.predictor.quantiles}
        for X_i, y_i in zip(X_cal, y_cal):
            scores_i = self.get_conformity_score(predictor=self.predictor, X_i=X_i, y_i=y_i)
            for alpha in self.predictor.quantiles:
                self.conformity_scores[alpha].append(scores_i[alpha])
        self._is_calibrated = True

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        quantile_preds = self.predictor.query([data])
        assert set(quantile_preds.keys()) == set(self.conformity_scores.keys()) == set(percentiles)
    
        quantiles = []
        for p in percentiles:
            pred = quantile_preds[p]
            if p < 0.5:
                target_p =  1 - p 
                correction = np.quantile(self.conformity_scores[p], target_p)
                quantile = pred - correction
            elif p > 0.5:
                target_p = p
                correction = np.quantile(self.conformity_scores[p], target_p)
                quantile = pred + correction
            else:
                quantile = pred
            quantiles.append(quantile)
        return ConditionalEstimation(point_prediction=quantile_preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))


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
        cv_splits = self._cross_val_split(X=X, y=y)

        self.conformity_scores = []
        self.fitted_predictors = []
        for X_train, X_cal, y_train, y_cal in cv_splits:
            predictor = copy.deepcopy(self.predictor)
            predictor.fit(X_train, y_train, loss=nn.L1Loss())
            self.fitted_predictors.append(predictor)
            
            for X_i, y_i in zip(X_cal, y_cal):
                self.conformity_scores.append(self.get_conformity_score(predictor=predictor, X_i=X_i, y_i=y_i))

        assert len(self.conformity_scores) == len(X)
        self._is_calibrated = True

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        preds, mean, std = [], [], []
        for predictor in self.fitted_predictors:     
            preds_j = np.squeeze(predictor.query([data]))
            preds.append(preds_j)
            mean.append(np.mean(preds_j))
            std.append(np.std(preds_j))
        # aggregate over folds
        mean = np.mean(mean)
        std = np.mean(std)

        quantiles = []
        for p in percentiles:
            n_cal = len(self.conformity_scores)
            if p < 0.5:
                target_p = min((1 - 2 * p) * (1 + 1 / n_cal), 1) # adjusted percentil for finite sample
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean - correction
            elif p > 0.5:
                target_p = min((2 * p - 1) * (1 + 1 / n_cal), 1)
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean + correction
            else:
                quantile = mean
            quantiles.append(quantile)
        return ConditionalEstimation(point_prediction=preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))

    

class QuantileCrossValCPCalibrator(CrossValCPMixin, QuantileCalibrationMixin, BaseCalibrator):
    """Uncertainty calibrator based on ensemble predictor and cross-validation based Conformal Prediction."""
                 
    def calibrate(self, data: tuple[list[Graph], list[float]]):
        X, y = data
        cv_splits = self._cross_val_split(X=X, y=y)

        self.fitted_predictors = []
        self.conformity_scores = {alpha: [] for alpha in self.predictor.quantiles}
        for X_train, X_cal, y_train, y_cal in cv_splits:
            predictor = copy.deepcopy(self.predictor)
            predictor.fit(X_train, y_train)
            self.fitted_predictors.append(predictor)
            
            for X_i, y_i in zip(X_cal, y_cal):
                scores_i = self.get_conformity_score(predictor=predictor, X_i=X_i, y_i=y_i)
                for alpha in self.predictor.quantiles:
                    self.conformity_scores[alpha].append(scores_i[alpha])
        self._is_calibrated = True

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        quantile_preds = {p: [] for p in percentiles}
        for predictor in self.fitted_predictors:     
            assert set(predictor.quantiles) == set(quantile_preds.keys())
            preds_j = predictor.query([data])
            for alpha, pred in preds_j.items():
                quantile_preds[alpha].append(preds_j[alpha])

        quantiles = []
        for p in percentiles:
            pred = np.mean(quantile_preds[p])
            if p < 0.5:
                target_p =  1 - p
                correction = np.quantile(self.conformity_scores[p], target_p)
                quantile = pred - correction
            elif p > 0.5:
                target_p = p
                correction = np.quantile(self.conformity_scores[p], target_p)
                quantile = pred + correction
            else:
                quantile = pred
            quantiles.append(quantile)
            
        return ConditionalEstimation(point_prediction=quantile_preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))
    

class EnsembleBootstrapCPCalibrator(EnsembleCalibrationMixin, BaseCalibrator):
    """Uncertainty calibrator based on ensemble predictor and split Conformal Prediction."""

    def _resample(self, X: list[Graph], y: list[float]) -> tuple[ArrayLike, list[Graph], list[float]]:
        """Sample with replacement from the full dataset. The size of the sampled set is identical to the given dataset."""
        indices = np.arange(len(X))
        bt_indices = resample(indices, replace=True, n_samples=len(X))
    
        bt_X, bt_y = [], []
        for i in bt_indices:
            bt_X.append(X[i])
            bt_y.append(y[i])
        return bt_indices, bt_X, bt_y

    def calibrate(self, data: tuple[list[Graph], list[float]]):
        """Get conformity scores using the calibration dataset."""
        X, y = data
        # fit boostrap models
        self.fitted_predictors = []
        predictors = self.predictor.get_ensemble()
        for predictor in predictors():
            assert isinstance(predictor, MLPPredictor)
            bootstrap_indices, bootstrapped_X, bootstrapped_y = self._resample(X=X, y=y)
            predictor.fit(bootstrapped_X, bootstrapped_y)
            predictor.indices = bootstrap_indices
            self.fitted_predictors.append(predictor)

        # compute conformity scores using
        conformity_scores = []
        for idx, (X_i, y_i) in enumerate(zip(X, y)):
            loo_preds = []   # leave-one-out prediction using all fitted models that are not fitted on (X_i, y_i)
            for predictor in self.fitted_predictors:
                if idx in predictor.indices:
                    continue
                loo_preds.append(predictor.query[X_i])
            if len(loo_preds) < 2:
                print(f"{idx}-th data point does not have sufficient predictions and is not included in any calibration set.") 
                continue

            mean = np.mean(loo_preds)
            std = np.std(loo_preds)
            conformity_scores.append(self.conformity_score_fn(value=y_i, mean=mean, std=std))
            self.conformity_scores = conformity_scores

    def get_conditional_estimation(self, data, percentiles = [0.05, 0.1, 0.5, 0.9, 0.95]) -> ConditionalEstimation:
        preds = np.squeeze([predictor.query([data]) for predictor in self.fitted_predictors])
        mean = np.mean(preds)
        std = np.std(preds)
    
        quantiles = []
        for p in percentiles:
            n_cal = len(self.conformity_scores)
            if p < 0.5:
                target_p = min((1 - 2 * p) * (1 + 1 / n_cal), 1) # adjusted percentil for finite sample
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean - correction
            elif p > 0.5:
                target_p = min((2 * p - 1) * (1 + 1 / n_cal), 1)
                correction = np.quantile(self.conformity_scores, target_p) * std
                quantile = mean + correction
            else:
                quantile = mean
            quantiles.append(quantile)
        return ConditionalEstimation(point_prediction=preds, distribution=PointwiseInterpolatedDist(values=(percentiles, np.array(quantiles))))      


def get_calibrator_class(predictor_type: PredictorType, calibrator_type: CalibratorType) -> Type[BaseCalibrator]:
    calibrator_map = {
        (PredictorType.ENSEMBLE_MLP, CalibratorType.GAUSSIAN): Gaussian,
        (PredictorType.ENSEMBLE_MLP, CalibratorType.CP_SPLIT): EnsembleSplitCPCalibrator,
        (PredictorType.ENSEMBLE_MLP, CalibratorType.CP_CROSSVAL): EnsembleCrossValCPCalibrator,
        (PredictorType.ENSEMBLE_MLP, CalibratorType.CP_BOOTSTRAP): EnsembleBootstrapCPCalibrator,
        (PredictorType.QUANTILE, CalibratorType.CP_SPLIT): QuantileSplitCPCalibrator,
        (PredictorType.QUANTILE, CalibratorType.CP_CROSSVAL): QuantileCrossValCPCalibrator
    }
    return calibrator_map[(predictor_type, calibrator_type)]