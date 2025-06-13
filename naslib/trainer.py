import time
import json
import logging
import os
import copy
import dill as pickle
from typing import Callable
import numpy as np
import codecs
from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
from naslib.config import FullConfig
from naslib.optimizers.base import MetaOptimizer
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.plot import plot_architectural_weights
from naslib.utils.tools import AverageMeter, Checkpointer, AttrDict

logger = logging.getLogger(__name__)



class Trainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """

    def __init__(self, optimizer: MetaOptimizer, config: FullConfig, lightweight_output=False):
        self.optimizer = optimizer
        self.config = config
        self.seed = config.seed
        self.save = config.save
        self.epochs = config.search.epochs
        self.lightweight_output = lightweight_output

        # preparations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # measuring stuff
        self.train_top1 = AverageMeter()
        self.train_top5 = AverageMeter()
        self.train_loss = AverageMeter()
        self.val_top1 = AverageMeter()
        self.val_top5 = AverageMeter()
        self.val_loss = AverageMeter()

        n_parameters = optimizer.get_model_size()
        self.search_trajectory = AttrDict(
            {
                "train_acc": [],
                "train_loss": [],
                "valid_acc": [],
                "valid_loss": [],
                "test_acc": [],
                "test_loss": [],
                "runtime": [],
                "train_time": [],
                "arch_eval": [],
                "params": n_parameters,
                "calibration_score": []
            }
        )

    def search(self, resume_from="", summary_writer=None, after_epoch: Callable[[int], None]=None, report_incumbent=True):
        """Start the architecture search and generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then train from scratch.
        """
        logger.info("Beginning search")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        checkpoint_freq = self.config.search.checkpoint_freq
        
        start_epoch = 0
        if checkpoint_freq is not None:
            os.makedirs(self.save + '/search', exist_ok=True)
            start_epoch = self._setup_checkpointers(save=self.save, resume_from=resume_from, period=checkpoint_freq)

        for e in range(start_epoch, self.epochs):

            start_time = time.time()
            self.optimizer.new_epoch(e)
            end_time = time.time()
            train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, train_time = self.optimizer.train_statistics()
            calibration_score = self.optimizer.calibration_score 

            self.search_trajectory.train_acc.append(train_acc)
            self.search_trajectory.train_loss.append(train_loss)
            self.search_trajectory.valid_acc.append(valid_acc)
            self.search_trajectory.valid_loss.append(valid_loss)
            self.search_trajectory.test_acc.append(test_acc)
            self.search_trajectory.test_loss.append(test_loss)
            self.search_trajectory.runtime.append(end_time - start_time)
            self.search_trajectory.train_time.append(train_time)
            self.search_trajectory.calibration_score.append(calibration_score)

            self.train_top1.avg = train_acc
            self.val_top1.avg = valid_acc

            if hasattr(self, "periodic_checkpointer"):
                self.periodic_checkpointer.step(e)

            self._log_to_json()
            self._log_and_reset_accuracies(e, summary_writer)

        with open(f'{self.save}/search_log.pt', "wb") as f:
            pickle.dump(self.optimizer.obs_and_condest, f)

        if self.config.save_arch_weights:
            logger.info(f"Saving architectural weight tensors: {self.save}/arch_weights.pt")
            best_arch = self.optimizer.get_final_architecture()
            arch_weights = best_arch.edges.data()
            with open(f'{self.save}/arch_weights.pt', "wb") as f:
                pickle.dump(best_arch, f)
                pickle.dump(arch_weights, f)
        if self.config.plot_arch_weights:
            plot_architectural_weights(arch_weights)

        logger.info("Training finished")

    def evaluate(
        self,
        retrain:bool=True,
        search_model:str="",
        resume_from:str="",
        best_arch:Graph=None,
        dataset_api:object=None,
        metric:Metric=None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that, otherwise train as defined in the config.

        Args:
            retrain (bool): Reset the weights from the architecure search
            search_model (str): Path to checkpoint file that was created during search. If not provided, then try to load 'model_final.pth' from search.
            resume_from (str): Resume retraining from the given checkpoint file.
            best_arch: Parsed model you want to directly evaluate and ignore the final model from the optimizer.
            dataset_api: Dataset API to use for querying model performance.
            metric: Metric to query the benchmark for.
        """
        logger.info("Start evaluation")
        if not best_arch:
            if not search_model:
                search_model = os.path.join(self.save, "search", "model_final.pth")
            self._setup_checkpointers(save=self.save, resume_from=search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if metric is None:
            metric = Metric.TEST_ACCURACY
        result = best_arch.query(
            metric=metric, dataset=self.config.dataset, dataset_api=dataset_api
        )
        logger.info("Queried results ({}): {}".format(metric, result))
        return result

    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train accuracy: {:.5f}, Validation accuracy: {:.5f}".format(
                epoch,
                self.train_top1.avg,
                self.val_top1.avg,
            )
        )

        if writer is not None:
            writer.add_scalar('Train accuracy (top 1)', self.train_top1.avg, epoch)
            writer.add_scalar('Train accuracy (top 5)', self.train_top5.avg, epoch)
            writer.add_scalar('Train loss', self.train_loss.avg, epoch)
            writer.add_scalar('Validation accuracy (top 1)', self.val_top1.avg, epoch)
            writer.add_scalar('Validation accuracy (top 5)', self.val_top5.avg, epoch)
            writer.add_scalar('Validation loss', self.val_loss.avg, epoch)

        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()

    def _setup_checkpointers(self, save: str, resume_from="", search=True, period=1, **add_checkpointables):
        """
        Sets up a periodic chechkpointer which can be used to save checkpoints
        at every epoch. It will call optimizer's `get_checkpointables()` as objects
        to store.

        Args:
            resume_from (str): A checkpoint file to resume the search or evaluation from.
            search (bool): Whether search or evaluation phase is checkpointed. This is required
                because the files are in different folders to not be overridden
            add_checkpointables (object): Additional things to checkpoint together with the
                optimizer's checkpointables.
        """
        checkpointables = self.optimizer.get_checkpointables()
        checkpointables.update(add_checkpointables)

        checkpointer = Checkpointer(
            model=checkpointables.pop("model"),
            save_dir=save + "/search" if search else save + "/eval",
            # **checkpointables #NOTE: this is throwing an Error
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=period,
            max_iter=self.epochs
        )

        if resume_from:
            logger.info("loading model from file {}".format(resume_from))
            checkpoint = checkpointer.resume_or_load(resume_from, resume=True)
            if checkpointer.has_checkpoint():
                return checkpoint.get("iteration", -1) + 1
        return 0

    def _log_to_json(self):
        """log training statistics to json file"""
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        if not self.lightweight_output:
            with codecs.open(
                os.path.join(self.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(self.search_trajectory, file, separators=(",", ":"))
        else:
            with codecs.open(
                os.path.join(self.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                lightweight_dict = copy.deepcopy(self.search_trajectory)
                for key in ["arch_eval", "train_loss", "valid_loss", "test_loss"]:
                    lightweight_dict.pop(key)
                json.dump([self.config, lightweight_dict], file, separators=(",", ":"))
