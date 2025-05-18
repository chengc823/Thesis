import pickle
import os
from typing import Literal, Protocol


class DataAPI(Protocol):
    
    def __call__(self, dataset: str):...


def get_nasbench201_api(dataset: Literal["cifar10", "cifar100", "ImageNet16-120", 'ninapro']):
    """Load the NAS-Bench-201 data."""

    datafiles = {
        'cifar10': 'nb201_cifar10_full_training.pickle',
        'cifar100': 'nb201_cifar100_full_training.pickle',
        'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle',
        'ninapro': 'nb201_ninapro_full_training.pickle'
    }
    datafile_path = os.path.join(os.getcwd(), 'naslib/data', datafiles[dataset])
    assert os.path.exists(datafile_path)

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)
    return {"nb201_data": data}