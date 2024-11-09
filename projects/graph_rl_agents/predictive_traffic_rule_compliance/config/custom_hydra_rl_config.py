from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from commonroad_geometric.learning.reinforcement.project.hydra_rl_config import RLProjectConfig


@dataclass
class CustomRLProjectConfig(RLProjectConfig):
    reactive_planner: Dict = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="custom_rl_config", node=CustomRLProjectConfig)
