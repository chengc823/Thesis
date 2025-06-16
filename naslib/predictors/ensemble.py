import torch.nn as nn
import numpy as np
import copy
from naslib.predictors.base import Predictor


class Ensemble(Predictor):
    def __init__(
        self,
        base_predictor: Predictor,
        num_ensemble=5,
        hpo_wrapper=True
    ):
        self.base_preidctor = base_predictor
        self.num_ensemble = num_ensemble
        self.hpo_wrapper = hpo_wrapper
        self.hyperparams = None
        self.ensemble = None
    
    def get_ensemble(self):
        return [
            copy.deepcopy(self.base_preidctor)
            for _ in range(self.num_ensemble)
        ]

    def fit(self, xtrain, ytrain, train_info=None, loss: nn.Module = nn.L1Loss()):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], "default_hyperparams"):
            self.hyperparams = self.ensemble[0].default_hyperparams.copy()
        self.set_hyperparams(self.hyperparams)

        train_errors = []
        for i in range(self.num_ensemble):
           train_error = self.ensemble[i].fit(xtrain, ytrain, train_info, loss=loss)
           train_errors.append(train_error)

        return train_errors

    def query(self, xtest, info=None):
        predictions = []
        for i in range(self.num_ensemble):
            prediction = self.ensemble[i].query(xtest, info)
            predictions.append(prediction)

        return np.array(predictions)

    def set_hyperparams(self, params):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            model.set_hyperparams(params)
        self.hyperparams = params