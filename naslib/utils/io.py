import yaml
import json
from typing import Type
import pydantic
import os


def read_config_from_yaml(filepath: str, config_type: Type[pydantic.BaseModel]) -> pydantic.BaseModel:
    with open(filepath, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
        config = config_type(**yaml_dict)
    return config


def dump_config_to_yaml(output_dir: str, config: pydantic.BaseModel):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/config.yml", "w") as f:
        yaml.dump(config.model_dump(), f, sort_keys=False)


def read_json(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
