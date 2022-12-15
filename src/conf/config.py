"""Classes to define structure and typing of the configuration objects."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Paths:
    """Class to manage data paths."""

    dataset: str


@dataclass
class Data:
    """Class to manage dataset attributes."""

    target: str
    text_feature: str
    positive_value: str


@dataclass
class Preprocess:
    """Class related to preprocessing parameters."""

    nb_split: int
    fraction_sample: float
    test_size: float
    seed: int
    regex_pattern_to_apply: Dict[str, str]


@dataclass
class Params:
    """Class related to models parameters."""

    logistic_reg: Dict[str, Any]


@dataclass
class ModelRegistry:
    """Class related to model registry settings."""

    uri: str


@dataclass
class ImbdConfig:
    """Global class config to work with the Imbd dataset."""

    paths: Paths
    data: Data
    preprocess: Preprocess
    params: Params
    ml_registry: ModelRegistry
