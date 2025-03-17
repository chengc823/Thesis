from enum import Enum
from pydantic import BaseModel, model_validator



class SearchConfig(BaseModel):
    predictor_type: str = "bananas"
    seed: int = 99
    epochs: int = 3

    checkpoint_freq: int = 5
    
    k: int = 10
    num_init: int = 1
    num_ensemble: int = 3
    acq_fn_type: str = "its"
    acq_fn_optimization: str = "mutation"
    encoding_type: str = "path"
    num_arches_to_mutate: int = 2
    max_mutations: int = 1
    num_candidates: int = 100

    gpu:  int | None = None


class EvalConfig(BaseModel):
    epochs: int = 2
   
    grad_clip: int = 5
    checkpoint_freq: int = 5

    drop_path_prob: float = 0.2
    auxiliary_weight: float = 0.4

    #: number of nodes for distributed training
    world_size: int = 1
    #: node rank for distributed training
    rank: int = 0
    #: url used to set up distributed training
    dist_url: str = "tcp://127.0.0.1:8888"
    #: distributed backend
    dist_backend: str = "nccl"
    multiprocessing_distributed: bool = True
    gpu:  int | None = None


class FullConfig(BaseModel):
    model_path: str | None  = None
    data_path: str | None = None
    #: Use multi-processing distributed training to launch N processes per node, which has N GPUs. 
    # This is the fastest way to use PyTorch for either single node or multi node data parallel training
    multiprocessing_distributed: bool = True
    #: GPU id to use
    gpu: int | None = None
    optimizer: str = "bananas"
    #: random seed
    seed: int = 99

    dataset: str = "cifar10"
    search_space: str = "nasbench201"

    search: SearchConfig = SearchConfig()
    #: perform evaluation only
    eval_only: bool = False
    evaluation: EvalConfig = EvalConfig()

    #: Resume from last checkpoint
    resume: bool = False

    #: Export
    #: Path to save the results
    out_dir: str = "run"
    save_arch_weights: bool = True
    plot_arch_weights: bool = False
    save: str | None = None
    opts: tuple | None = None

    @model_validator(mode="after")
    def process(self):
        self.search.seed = self.seed
        self.evaluation.multiprocessing_distributed = self.multiprocessing_distributed
        self.search.gpu = self.evaluation.gpu = self.gpu

        self.save = f"{self.out_dir}/{self.search_space}/{self.dataset}/{self.optimizer}/{self.seed}"
        self.opts = None
        return self
    