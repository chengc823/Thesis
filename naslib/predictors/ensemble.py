import numpy as np
import copy
from naslib.predictors.base import Predictor


class Ensemble(Predictor):
    def __init__(
        self,
        base_predictor: Predictor,
        num_ensemble=5,
        hpo_wrapper=True,
        zc=None,
        zc_only=None
    ):
        self.base_preidctor = base_predictor
        self.num_ensemble = num_ensemble
        self.hpo_wrapper = hpo_wrapper
        self.hyperparams = None
        self.ensemble = None
        self.zc = zc
        self.zc_only = zc_only

    def get_ensemble(self):
        return [
            copy.deepcopy(self.base_preidctor)
            for _ in range(self.num_ensemble)
        ]

    def fit(self, xtrain, ytrain, train_info=None):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], "default_hyperparams"):
            self.hyperparams = self.ensemble[0].default_hyperparams.copy()
        self.set_hyperparams(self.hyperparams)

        train_errors = []
        for i in range(self.num_ensemble):
            train_error = self.ensemble[i].fit(xtrain, ytrain, train_info)
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

    def set_random_hyperparams(self):
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        if self.hyperparams is None and hasattr(self.ensemble[0], "default_hyperparams"):
            params = self.ensemble[0].default_hyperparams.copy()
            
        params = self.hyperparams or self.ensemble[0].set_random_hyperparams()
        self.set_hyperparams(params)
        return params

    def set_pre_computations(
        self,
        unlabeled=None,
        xtrain_zc_info=None,
        xtest_zc_info=None,
        unlabeled_zc_info=None,
    ):
        """
        Some predictors have pre_computation steps that are performed outside the
        predictor. E.g., omni needs zerocost metrics computed, and unlabeled data
        generated. In the case of an ensemble, this method relays that info to
        the predictor.
        """
        if self.ensemble is None:
            self.ensemble = self.get_ensemble()

        for model in self.ensemble:
            assert hasattr(
                model, "set_pre_computations"
            ), "set_pre_computations() not implemented"
            model.set_pre_computations(
                unlabeled=unlabeled,
                xtrain_zc_info=xtrain_zc_info,
                xtest_zc_info=xtest_zc_info,
                unlabeled_zc_info=unlabeled_zc_info,
            )
