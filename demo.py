import numpy as np
from tqdm import tqdm
from naslib.utils.io import read_config_from_yaml
from naslib.config import FullConfig
from naslib.trial import trial

file_path = "/Users/chengchen/GitHub/master_thesis/notebooks/experiment.yml"
config = read_config_from_yaml(filepath=file_path, config_type=FullConfig)

TEST = True
output_dir = "/Users/chengchen/Desktop/Test" if TEST else None

NUM_TRIALS = 1
START_TRIAL = 20
sampled_seeds = np.arange(START_TRIAL, START_TRIAL + NUM_TRIALS)
for seed in tqdm(sampled_seeds, total=NUM_TRIALS):
    seed = int(seed)
    trial(config=config, seed=seed, dump_config=True, output_dir=output_dir)