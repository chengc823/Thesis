from naslib.config import SearchSpaceType, FullConfig
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.data.api import get_nasbench201_api
from naslib.trainer import Trainer
from naslib.optimizers.bananas.optimizer import Bananas
from naslib.utils.io import dump_config_to_yaml



def get_search_space(search_space_type: SearchSpaceType) -> Graph:
    MAP = {
        SearchSpaceType.NB201: NasBench201SearchSpace()
    }
    return MAP[search_space_type]


def get_dataset_api(search_space_type: SearchSpaceType, dataset: str):
    MAP = {
        SearchSpaceType.NB201: get_nasbench201_api
    }
    return MAP[search_space_type](dataset=dataset)



def trial(config: FullConfig, seed=42, output_dir: str | None = None, dump_config: bool = True):
    # Overwrite random seed and dump the config file to the output dir
    config.seed = seed
    if output_dir:
        config.out_dir = output_dir
    if dump_config:
        dump_config_to_yaml(output_dir=config.save, config=config)
    # Get seach space
    search_space = get_search_space(search_space_type=config.search_space)
    # Setup optimizer and trainer
    optimizer = Bananas(config=config)
    dataset_api = get_dataset_api(search_space_type=config.search_space, dataset=config.dataset)
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)
    trainer = Trainer(optimizer=optimizer, config=config)
    # Search
    trainer.search()
    # Evaluate
    if config.search.checkpoint_freq:
        performance = trainer.evaluate(dataset_api=dataset_api)
    else:
        performance = trainer.evaluate(dataset_api=dataset_api, best_arch=optimizer.get_final_architecture())
    print(f"seed={seed}: {performance}")  