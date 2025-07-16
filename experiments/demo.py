import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from naslib.utils.io import read_config_from_yaml
from naslib.config import FullConfig
from naslib.trial import trial

file_path = "/Users/chengchen/GitHub/master_thesis/experiments/config.yml"
config = read_config_from_yaml(filepath=file_path, config_type=FullConfig)

TEST = True
output_dir = f"/Users/chengchen/Desktop/Experiments/acq_search={config.search.acq_fn_optimization}" if not TEST else "/Users/chengchen/Desktop/test"

NUM_TRIALS = 19
START_TRIAL = 31
sampled_seeds = np.arange(START_TRIAL, START_TRIAL + NUM_TRIALS)


for seed in tqdm(sampled_seeds, total=NUM_TRIALS):
    seed = int(seed)
    trial(config=config, seed=seed, dump_config=True, output_dir=output_dir)



# file_dir = "./experiments/configs"
# TEST = False
# output_dir = f"/Users/chengchen/Desktop/Experiments/acq_search={config.search.acq_fn_optimization}" if not TEST else "/Users/chengchen/Desktop/test"

# NUM_TRIALS = 25
# START_TRIAL = 25
# sampled_seeds = np.arange(START_TRIAL, START_TRIAL + NUM_TRIALS)

# for f in os.listdir(file_dir):
#     f = os.path.join(file_dir, f)
#     config = read_config_from_yaml(filepath=f, config_type=FullConfig)

#     for seed in tqdm(sampled_seeds, total=NUM_TRIALS):
#         seed = int(seed)
#         trial(config=config, seed=seed, dump_config=True, output_dir=output_dir)
