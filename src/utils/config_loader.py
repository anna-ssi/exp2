from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    type: str
    size: int
    hidden: int


@dataclass
class OptimConfig:
    type: str
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01


class ConfigLoader:
    def __init__(self, config: Dict[str, Any]):
        self.seed = config['seed']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.test_size = config['test_size']

        self.network = NetworkConfig(**config['network'])
        self.optim = OptimConfig(**config['optim'])

    @staticmethod
    def load_network_config(config):
        return NetworkConfig(**config)

    @staticmethod
    def load_optim_config(config):
        return OptimConfig(**config)
