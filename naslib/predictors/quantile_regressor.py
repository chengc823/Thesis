import copy
import torch
import torch.nn as nn
from naslib.predictors.base import Predictor


class PinBallLoss(nn.Module):
    def __init__(self, alpha: float, reduction='sum'):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("Quantile should be in (0, 1) range")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        errors = target - input
        loss = torch.where(errors > 0, self.alpha * errors, (1 - self.alpha) * (-errors))
        return loss.sum()


class QuantileRegressor(Predictor):

    def __init__(self, base_predictor: Predictor, quantiles: list[float]):
        self.base_preidctor = base_predictor
        self.quantiles = quantiles
        self.quantile_regressors = {quantile: copy.deepcopy(self.base_preidctor) for quantile in quantiles}

    def fit(self, xtrain, ytrain, train_info=None):
        train_errors = {}
        for quantile in self.quantiles:
           estimator = self.quantile_regressors[quantile]
           train_error = estimator.fit(xtrain, ytrain, train_info, loss=PinBallLoss(alpha=quantile))
           self.quantile_regressors[quantile] = estimator
           train_errors[quantile]= train_error

        return train_errors

    def query(self, xtest, info=None):
        predictions = {}
        for quantile, estimator in self.quantile_regressors.items():
            predictions[quantile] = estimator.query(xtest, info)

        return predictions





