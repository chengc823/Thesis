# import collections
import logging
import torch
# import copy
import numpy as np
from naslib.optimizers.base import MetaOptimizer
import naslib.optimizers.bananas.acquisition_functions as acq
from naslib.optimizers.bananas.calibrator import get_calibrator_class, calibration_metrics
from naslib.optimizers.bananas.distribution import get_quantile_levels
from naslib.predictors.base import Predictor
from naslib.predictors.ensemble import Ensemble
from naslib.predictors.mlp import MLPPredictor
# from naslib.predictors.zerocost import ZeroCost
# from naslib.predictors.utils.encodings import encode_spec
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.config import FullConfig, PredictorType, EncodingType, ACQType
from naslib.utils.tools import count_parameters_in_MB # , get_train_val_loaders, AttrDict
# from naslib.utils.log import log_every_n_seconds


logger = logging.getLogger(__name__)



def _get_predictor(predictor_type: PredictorType, encoding_type: EncodingType, **kwargs) -> Predictor:

    match predictor_type:
        case PredictorType.ENSEMBLE_MLP:
            predictor = Ensemble(base_predictor=MLPPredictor(encoding_type=encoding_type), **kwargs)

        case PredictorType.MLP:
            predictor = MLPPredictor(encoding_type=encoding_type, **kwargs)
    
                
             #   num_ensemble=self.num_ensemble, ss_type=self.ss_type, encoding_type=self.encoding_type)#,
                            #  predictor_type=self.predictor_type,
                            #  zc=self.zc,
                            #  zc_only=self.zc_only,
                               # config=self.config)
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
        #   self.epochs = config.search.epochs
     #   self.num_ensemble = config.search.num_ensemble
        # self.predictor_type = config.search.predictor_type

        # Surrogate model
        self.encoding_type = config.search.encoding_type     # # currently not implemented
        self.predictor_type = config.search.predictor_type
        self.predictor_params = config.search.predictor_params
        self.performance_metric = Metric.VAL_ACCURACY # The metric score for training surrogate model
        # Calibrator
        self.calibrator_type = config.search.calibrator_type
        self.train_cal_split = config.search.train_cal_split
        self.calibrator_params = config.search.calibrator_params 
        self.percentiles = get_quantile_levels(num_quantiles=config.search.num_quantiles)

        # Acquisition functions
        # define an acquisition function
     #   self.acq_fn = get_acquisition_function(acq_type=self.acq_fn_type, **self.acq_fn_params)
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_params = config.search.acq_fn_params
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.num_candidates = config.search.num_candidates
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
      #  self.max_zerocost = 1000

        # a container storing evaluated models. training data (model_encoding, model_metrics) for fitting the surrogate model 
        # can be built from train_data using self._get_train
        self.train_data = []
        self.conditional_estimations = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        # self.zc = config.search.zc if hasattr(config.search, 'zc') else None
        # self.semi = "semi" in self.predictor_type 
        # self.zc_api = zc_api
        # self.use_zc_api = config.search.use_zc_api if hasattr(
        #     config.search, 'use_zc_api') else False
        # self.zc_names = config.search.zc_names if hasattr(
        #     config.search, 'zc_names') else None
        # self.zc_only = config.search.zc_only if hasattr(
        #     config.search, 'zc_only') else False
        
        # self.load_labeled = config.search.load_labeled if hasattr(
        #     config.search, 'load_labeled') else False

    def adapt_search_space(self, search_space: Graph, scope=None, dataset_api=None):
        assert (search_space.QUERYABLE), "Bananas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        self.ss_metrics_mapping = self.search_space.METRIC_TO_SEARCH_SPACE
        # if self.zc:
        #     self.train_loader, _, _, _, _ = get_train_val_loaders(
        #         self.config, mode="train")
        # if self.semi:
        #     self.unlabeled = []

    # def get_zero_cost_predictors(self):
    #     return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}

    # def query_zc_scores(self, arch):
    #     zc_scores = {}
    #     zc_methods = self.get_zero_cost_predictors()
    #     arch_hash = arch.get_hash()
    #     for zc_name, zc_method in zc_methods.items():

    #         if self.use_zc_api and str(arch_hash) in self.zc_api:
    #             score = self.zc_api[str(arch_hash)][zc_name]['score']
    #         else:
    #             zc_method.train_loader = copy.deepcopy(self.train_loader)
    #             score = zc_method.query(arch, dataloader=zc_method.train_loader)

    #         if float("-inf") == score:
    #             score = -1e9
    #         elif float("inf") == score:
    #             score = 1e9

    #         zc_scores[zc_name] = score

    #     return zc_scores

    def _set_scores(self, model):

        # if self.use_zc_api and str(model.arch_hash) in self.zc_api:
        #     model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']
        # else:
        model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)

        # if self.zc and len(self.train_data) <= self.max_zerocost:
        #     model.zc_scores = self.query_zc_scores(model.arch)

        self.train_data.append(model)
        self._update_history(model)

    def _sample_new_model(self) -> torch.nn.Module:
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=False)#self.load_labeled)
        model.arch_hash = model.arch.get_hash()
        
        if self.search_space.instantiate_model == True:
            model.arch.parse()
        return model

    def _get_data(self):
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    # def _get_predictor(self) -> Predictor:
    #     if self.predictor_type == PredictorType.ENSEMBLE_MLP:
    #         predictor = Ensemble(base_predictor=MLPPredictor(encoding_type=self.encoding_type), **self.predictor_params)
                
    #          #   num_ensemble=self.num_ensemble, ss_type=self.ss_type, encoding_type=self.encoding_type)#,
    #                         #  predictor_type=self.predictor_type,
    #                         #  zc=self.zc,
    #                         #  zc_only=self.zc_only,
    #                            # config=self.config)
    #     return predictor

    def _get_new_candidates(self, ytrain) -> list[torch.nn.Module]:
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == 'random_sampling':
            for _ in range(self.num_candidates):
                # self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api) # # FIXME extend to Zero Cost case
                model = self._sample_new_model()
             #   model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
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

                # if self.semi:
                #     # create unlabeled data and pass it to the predictor
                #     while len(self.unlabeled) < len(xtrain):
                #         model = self._sample_new_model()

                #         # if self.zc and len(self.train_data) <= self.max_zerocost:
                #         #     model.zc_scores = self.query_zc_scores(model.arch)

                #         self.unlabeled.append(model)

                #     ensemble.set_pre_computations(
                #         unlabeled=[m.arch for m in self.unlabeled]
                #     )

                # if self.zc and len(self.train_data) <= self.max_zerocost:
                #     # pass the zero-cost scores to the predictor
                #     train_info = {'zero_cost_scores': [
                #         m.zc_scores for m in self.train_data]}
                #     ensemble.set_pre_computations(xtrain_zc_info=train_info)

                #     if self.semi:
                #         unlabeled_zc_info = {'zero_cost_scores': [
                #             m.zc_scores for m in self.unlabeled]}
                #         ensemble.set_pre_computations(
                #             unlabeled_zc_info=unlabeled_zc_info)
               
                # predictor.fit(xtrain, ytrain)
                # get a calibrated distribution (CDF)
               # calibrator = get_calibrator(calibrator_type=self.calibrator_type, predictor=predictor)
                
                # get candidates for next exploration 
                candidates = self._get_new_candidates(ytrain=y)
    
               # acq_fn = acquisition_function(distribution=distribution, threshold=max(ytrain), acq_type=self.acq_fn_type,  **self.acq_fn_params)
              ## # acquisition_function(predictor=predictor, ytrain=ytrain, acq_fn_type=self.acq_fn_type, **self.acq_fn_params)

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
                self.next_batch = [candidates[i] for i in sorted_indices[-self.k:]]#self._get_best_candidates(candidates, acq_fn)
                self.next_batch_estimations = [conditional_estimation[i] for i in sorted_indices[-self.k:]]

            # train the next architecture chosen by the neural predictor
            # add model to train_data 
            self._set_scores(self.next_batch.pop()) 
            # add distribution conditional on the next archtecture into list
            self.conditional_estimations.append(self.next_batch_estimations.pop())
            # compute calibration score
            self.obs_and_condest = list(zip(self._get_data()[1], self.conditional_estimations))
            self.calibration_score = calibration_metrics(obs_and_condest=self.obs_and_condest[self.num_init:], percentiles=self.percentiles)

              
    # def _get_best_candidates(self, candidates: list[torch.nn.Module]):
    #     # if self.zc and len(self.train_data) <= self.max_zerocost:
    #     #     for model in candidates:
    #     #         model.zc_scores = self.query_zc_scores(model.arch)

    #     #     values = [acq_fn(model.arch, [{'zero_cost_scores': model.zc_scores}]) for model in candidates]
    #     # else:
    #     values = [acq_fn(model.arch) for model in candidates]

    #     sorted_indices = np.argsort(values)
    #     choices = [candidates[i] for i in sorted_indices[-self.k:]]

    #     return choices

    def _update_history(self, child):
        #if len(self.history) < 100:
        self.history.append(child)
        # else:
        #     for i, p in enumerate(self.history):
        #         if child.accuracy > p.accuracy:
        #             self.history[i] = child
        #             break

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



        




        # train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, train_time
        
        # if self.search_space.space_name != "nasbench301":
        #     return (
        #         best_arch.query(
        #             Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
        #         ),
        #         best_arch.query(
        #             Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
        #         ),
        #         best_arch.query(
        #             Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
        #         ),
        #         best_arch.query(
        #             Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
        #         ),
        #     )
        # else:
        #     return (
        #         -1, 
        #         best_arch.query(
        #             Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
        #         ),
        #         best_arch.query(
        #             Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
        #         ),
        #         best_arch.query(
        #             Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
        #         ),
        #     ) 

    def test_statistics(self):
        if Metric.RAW in self.ss_metrics_mapping:
            best_arch = self.get_final_architecture()
        #if self.search_space.space_name != "nasbench301":
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

    # def get_arch_as_string(self, arch):
    #     if self.search_space.get_type() == 'nasbench301':
    #         str_arch = str(list((list(arch[0]), list(arch[1]))))
    #     else:
    #         str_arch = str(arch)
    #     return str_arch
