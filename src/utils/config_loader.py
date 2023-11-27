from typing import Dict, Any
from dataclasses import dataclass


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
        self.test_size = config.get('test_size', 0.1)
        self.net_type = config.get('net_type', None)
        self.kfold = config.get('kfold', 5)

        self.optim = OptimConfig(**config['optim'])

    @staticmethod
    def load_optim_config(config):
        return OptimConfig(**config)
