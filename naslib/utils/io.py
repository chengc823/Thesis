import yaml
from typing import Type
import pydantic


def read_yaml(filepath: str, config_type: Type[pydantic.BaseModel]) -> pydantic.BaseModel:
    with open(filepath, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
        config = config_type(**yaml_dict)
    return config


def dump_yaml(filepath: str, config: pydantic.BaseModel):
    with open(filepath, "w") as f:
        yaml.dump(config.model_dump(), f, sort_keys=False)
