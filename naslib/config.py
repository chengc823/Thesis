from enum import Enum
from pydantic import BaseModel, model_validator, Field
from typing import Literal, Any



class SearchSpaceType(str, Enum):
    NB201 = "nasbench201"


class EncodingType(str, Enum):
    """Architecture encoding type."""

    ADJACENCY_ONE_HOT = "adjacency_one_hot"
    ADJACENCY_MIX = 'adjacency_mix'
    PATH = "path"
    GCN = "gcn"
    BONAS = "bonas"
    SEMINAS = "seminas"
    COMPACT = 'compact'


class PredictorType(str, Enum):
    MLP = "mlp"
    ENSEMBLE_MLP = "ensemble_mlp"
    QUANTILE = "quantile"


class CalibratorType(str, Enum):  
    #: Default calibrator used in BANANAS (no calibration is peformed)
    GAUSSIAN = "gaussian"
    #: Calibrators based on Conformal Predictions
    CP_SPLIT = "CP_split"
    CP_CROSSVAL = "CP_cv"
    CP_QUANTILE = "CP_quantile"
    CP_BOOSTING = "CP_boosting"
    CP_BOOSTING_NBHD = "CP_boosting_NBHD"


class ACQType(str, Enum):
    ITS = "its"
    UCB = "ucb"
    PI = "pi"
    EI = "ei"
    EXPLOIT_ONLY = "exploit_only"




class NASConfig(BaseModel):
    # #: Random seed for search
    # seed: int = 99
    #: Whether or not and the frequency to setup checkpoints 
    checkpoint_freq: int | None = None
    #: Number of architectures being sampled and evaluated
    epochs: int = 150
    #: Architecture encoding type
    encoding_type: EncodingType = EncodingType.PATH
    
    # BO-related parameters
    #: Number of initially sampled data points before fitting the surrogate model for the first time
    num_init: int = 10
    
    #: Surrogate model
    predictor_type: PredictorType = PredictorType.ENSEMBLE_MLP
    predictor_params: dict[str, Any] 
    
    #: Calibrator: 
    num_quantiles: int = 20 
    train_cal_split: float | int = 0.3
    calibrator_type: CalibratorType = CalibratorType.GAUSSIAN
    calibrator_params: dict[str, Any]
    
    #: Acquisition functions
    acq_fn_type: ACQType = ACQType.ITS
    acq_fn_params: dict[str, Any]
    acq_fn_optimization: Literal["random_sampling", "mutation"] = "mutation"
    #: Mutation parameters (only relevant if "mutation" is used)
    #: the number of best ever-found models to be mutated
    num_arches_to_mutate: int = 2
    #: maximal mutation allowed for each model to get mutated
    max_mutations: int = 1
    #: The number of archtectures with which the acquisition function is called (Since it's not possible to run acq over the entire search space)
    num_candidates: int = 100
    #: Among the above picked candidates, the number of architectures picked to evaluate in parallel 
    # (i.e., number of candidates picked by acquisition function before refitting the surrogate model) 
    k: int = 10
    
    @model_validator(mode="before")
    def set_empty_params(cls, values):
        fields = ["predictor_params", "calibrator_params", "acq_fn_params"]
        for field in fields:
            if values[field] is None:
                values[field] = {}
        return values
    

   
    
    
    
   

   

    # gpu:  int | None = None


# class EvalConfig(BaseModel):
 #   epochs: int = 2
   
  #  grad_clip: int = 5
 #   checkpoint_freq: int = 5

 #   drop_path_prob: float = 0.2
  #  auxiliary_weight: float = 0.4

    #: number of nodes for distributed training
  #  world_size: int = 1
    #: node rank for distributed training
  #  rank: int = 0
    #: url used to set up distributed training
  #  dist_url: str = "tcp://127.0.0.1:8888"
    #: distributed backend
   # dist_backend: str = "nccl"
 #   multiprocessing_distributed: bool = True
  #  gpu:  int | None = None


class FullConfig(BaseModel):
   # model_path: str | None  = None
   # data_path: str | None = None
    #: Use multi-processing distributed training to launch N processes per node, which has N GPUs. 
    # This is the fastest way to use PyTorch for either single node or multi node data parallel training
    multiprocessing_distributed: bool = True
    #: GPU id to use
    gpu: int | None = None
    optimizer: str = "bananas"
    #: random seed
    seed: int 

    search_space: SearchSpaceType = SearchSpaceType.NB201
    dataset: Literal["cifar10", "cifar100", "ImageNet16-120"]

    search: NASConfig 
    # #: perform evaluation only
    # eval_only: bool = False
    # evaluation: EvalConfig = EvalConfig()

    #: Resume from last checkpoint
    resume: bool = False

    #: Export
    #: Path to save the results
    out_dir: str 
    save_arch_weights: bool = True
    plot_arch_weights: bool = False
 #  save: str | None = None
  #  opts: tuple | None = None

    @property
    def save(self):
      #  self.search.seed = self.seed
      #  self.evaluation.multiprocessing_distributed = self.multiprocessing_distributed
      #  self.search.gpu = self.gpu

      #  self.save = 
       # self.opts = None
        algo_base = f"{self.optimizer}__{self.search.predictor_type.value}__{self.search.calibrator_type.value}"
        match self.search.calibrator_type:
            case CalibratorType.GAUSSIAN:
                algo = algo_base + f"__num_quantiles={self.search.num_quantiles}"
            case _:
                algo = algo_base + f"__train_cal_split={self.search.train_cal_split}__num_quantiles={self.search.num_quantiles}".replace(".", "")

        full_path = f"{self.out_dir}/{self.search_space}/{self.dataset}/acq={self.search.acq_fn_type.value}/{algo}/num_init={self.search.num_init}/seed={self.seed}"
        return full_path
    
    def info(self) -> str:
      return f"{self.search_space}_{self.dataset}"

    