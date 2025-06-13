
import logging
import torch
import numpy as np
from naslib.optimizers.base import MetaOptimizer
import naslib.optimizers.bananas.acquisition_functions as acq
from naslib.optimizers.bananas.calibrator import get_calibrator_class
from naslib.optimizers.bananas.calibration_utils import rmsce_calibration
from naslib.optimizers.bananas.distribution import get_quantile_levels
from naslib.predictors.base import Predictor
from naslib.predictors.ensemble import Ensemble
from naslib.predictors.mlp import MLPPredictor
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.config import FullConfig, PredictorType, EncodingType, ACQType
from naslib.utils.tools import count_parameters_in_MB 


logger = logging.getLogger(__name__)



def _get_predictor(predictor_type: PredictorType, encoding_type: EncodingType, **kwargs) -> Predictor:

    match predictor_type:
        case PredictorType.ENSEMBLE_MLP:
            predictor = Ensemble(base_predictor=MLPPredictor(encoding_type=encoding_type), **kwargs)
        case PredictorType.QUANTILE:
            # TODO
            ...
    return predictor



class Bananas(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False

    def __init__(self, config: FullConfig, zc_api=None):
        super().__init__()
        self.config = config
        self.seed = config.seed
        self.dataset = config.dataset
        self.k = config.search.k
        self.num_init = config.search.num_init

        # Surrogate model
        self.encoding_type = config.search.encoding_type  
        self.predictor_type = config.search.predictor_type
        self.predictor_params = config.search.predictor_params
        self.performance_metric = Metric.VAL_ACCURACY # The metric score used as y to train surrogate model
        # Calibrator
        self.calibrator_type = config.search.calibrator_type
        self.train_cal_split = config.search.train_cal_split
        self.calibrator_params = config.search.calibrator_params 
        self.percentiles = get_quantile_levels(num_quantiles=config.search.num_quantiles)

        # Acquisition functions
        # define an acquisition function
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_params = config.search.acq_fn_params
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.num_candidates = config.search.num_candidates
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations

        # a container storing evaluated models. training data (model_encoding, model_metrics) for fitting the surrogate model 
        # can be built from train_data using self._get_train
        self.train_data = []
        self.conditional_estimations = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

    def adapt_search_space(self, search_space: Graph, scope=None, dataset_api=None):
        assert (search_space.QUERYABLE), "Bananas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        self.ss_metrics_mapping = self.search_space.METRIC_TO_SEARCH_SPACE

    def _set_scores(self, model):
        model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
        self.train_data.append(model)
        self._update_history(model)

    def _sample_new_model(self) -> torch.nn.Module:
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=False)
        model.arch_hash = model.arch.get_hash()
        
        if self.search_space.instantiate_model == True:
            model.arch.parse()
        return model

    def _get_data(self):
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    def _get_new_candidates(self, ytrain) -> list[torch.nn.Module]:
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == 'random_sampling':
            for _ in range(self.num_candidates):
                model = self._sample_new_model()
                candidates.append(model)

        elif self.acq_fn_optimization == 'mutation':
            # mutate the k best architectures by x
            best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
            best_archs = [self.train_data[i].arch for i in best_arch_indices]
            candidates = []
            for arch in best_archs:
                for _ in range(int(self.num_candidates / len(best_archs) / self.max_mutations)):    # number of candidates each selected best arch should derive
                    candidate = arch.clone()
                    for __ in range(int(self.max_mutations)):
                        arch = self.search_space.clone()
                        arch.mutate(candidate, dataset_api=self.dataset_api)
                        if self.search_space.instantiate_model == True:
                            arch.parse()
                        candidate = arch

                    model = torch.nn.Module()
                    model.arch = candidate
                    model.arch_hash = candidate.get_hash()
                    candidates.append(model)

        else:
            logger.info(f'{self.encoding_type} is not yet supported as a acq fn optimizer.')
            raise NotImplementedError()

        return candidates

    def new_epoch(self, epoch):
        if epoch < self.num_init:
            model = self._sample_new_model()
            self._set_scores(model)

            self.conditional_estimations.append(None)
            self.calibration_score = np.nan
        else:
            if len(self.next_batch) == 0:
                
                print(f"TRAINING surrogate predictor for epoch={epoch}.")
                # train and calibrate a surrogate model
                X, y = self._get_data()
                predictor = _get_predictor(predictor_type=self.predictor_type, encoding_type=self.encoding_type, **self.predictor_params)
                calibrator = get_calibrator_class(calibrator_type=self.calibrator_type)(
                    predictor=predictor, train_cal_split=self.train_cal_split, seed=self.seed, **self.calibrator_params
                )
                calibrator.calibrate(data=(X, y))

                # get candidates for next exploration 
                candidates = self._get_new_candidates(ytrain=y)
                # optimize the acquisition function to output k new architectures
                acq_values = []
                conditional_estimation = []
                for model in candidates:
                    model_condest = calibrator.get_conditional_estimation(data=model.arch, percentiles=self.percentiles)
                    distribution = model_condest.distribution       
                    # get acquisition score based on function type
                    match self.acq_fn_type:
                        case ACQType.ITS:
                            acq_score = acq.idependent_thompson_sampling(distribution=distribution, **self.acq_fn_params)
                        case ACQType.UCB:
                            acq_score = acq.upper_confidence_bound(distribution=distribution, **self.acq_fn_params)
                        case ACQType.PI | ACQType.EI:
                            acq_score = acq.probability_of_improvement(distribution=distribution, threhold=max(y), **self.acq_fn_params)
                    acq_values.append(acq_score)
                    conditional_estimation.append(model_condest)

                sorted_indices = np.argsort(acq_values)
                self.next_batch = [candidates[i] for i in sorted_indices[-self.k:]]
                self.next_batch_estimations = [conditional_estimation[i] for i in sorted_indices[-self.k:]]

            # train the next architecture chosen by the neural predictor
            # add model to train_data 
            self._set_scores(self.next_batch.pop()) 
            # add distribution conditional on the next archtecture into list
            self.conditional_estimations.append(self.next_batch_estimations.pop())
            # compute calibration score when there are more obs than the number of quantiles
            self.obs_and_condest = list(zip(self._get_data()[1], self.conditional_estimations))
            calibration_score = np.nan
            if len(self.obs_and_condest) > self.num_init + len(self.percentiles):
                calibration_score = rmsce_calibration(obs_and_condest=self.obs_and_condest[self.num_init:], percentiles=self.percentiles)
            self.calibration_score = calibration_score


    def _update_history(self, child):
        self.history.append(child)

    def train_statistics(self, report_incumbent=True):
        best_arch = self.get_final_architecture() if report_incumbent else self.train_data[-1].arch

        metrics_to_query = [
            Metric.TRAIN_ACCURACY, 
            Metric.TRAIN_LOSS,
            Metric.VAL_ACCURACY,
            Metric.VAL_LOSS,
            Metric.TEST_ACCURACY,
            Metric.TEST_LOSS,
            Metric.TRAIN_TIME
        ]

        statistics = [-1] * len(metrics_to_query)   # -1 is the default value if a metrics is not available
        for idx, metric in enumerate(metrics_to_query):
            if metric in self.ss_metrics_mapping:
                statistics[idx] =  best_arch.query(metric, self.dataset, dataset_api=self.dataset_api)
                
        return statistics

    def test_statistics(self):
        if Metric.RAW in self.ss_metrics_mapping:
            best_arch = self.get_final_architecture()
            return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        else:
            return -1

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)
