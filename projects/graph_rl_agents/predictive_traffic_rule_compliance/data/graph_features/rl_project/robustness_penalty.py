import numpy as np
from typing import Dict, Optional

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.common.vehicle import ControlledVehicle
from enum import Enum

class RobustnessType(Enum):
    G1 = "R_G1"
    G2 = "R_G2"
    G3 = "R_G3"
    G4 = "R_G4"
    I1 = "R_I1"
    I2 = "R_I2"
    I3 = "R_I3"
    I4 = "R_I4"
    I5 = "R_I5"


class BaseRobustnessPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
        robustness_type: RobustnessType
    ) -> None:
        self._weight = weight
        self._robustness_type = robustness_type
        self._rule_monitors: Dict[int, RuleEvaluator] = {}
        self._cumulative = 0.0
        self._cumulative_abs = 0.0
        self._min = float('inf')
        self._max = float('-inf')
        self._call_count = 0
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: Dict
    ) -> float:
        # ego_vehicle = simulation.ego_vehicle
        ego_vehicle = simulation.simulation.world_state.vehicle_by_id(-1)
        robustness = self._get_robustness(simulation, ego_vehicle)
        penalty = robustness * self._weight
        return penalty

    def _get_robustness(self, simulation: EgoVehicleSimulation, ego_vehicle: ControlledVehicle) -> float:
        if -1 not in self._rule_monitors:
            self._rule_monitors[-1] = RuleEvaluator.create_from_config(
                simulation.simulation.world_state,
                ego_vehicle,
                self._robustness_type.value
            )

        evaluator = self._rule_monitors[-1]
        robustness_series = evaluator.evaluate()
        return robustness_series[-1]

    def compute(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: Dict
    ) -> float:
        reward = self(action, simulation, data, observation)
        self._cumulative += reward
        self._cumulative_abs += abs(reward)
        self._min = min(self._min, reward)
        self._max = max(self._max, reward)
        self._call_count += 1
        return reward

    def reset(self, simulation: Optional[EgoVehicleSimulation] = None) -> None:
        self._rule_monitors.clear()
        self._cumulative = 0.0
        self._cumulative_abs = 0.0
        self._min = float('inf')
        self._max = float('-inf')
        self._call_count = 0

class G1RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.G1)

class G2RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.G2)

class G3RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.G3)

class G4RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.G4)

class I1RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.I1)

class I2RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.I2)

class I3RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.I3)

class I4RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.I4)

class I5RobustnessPenaltyRewardComputer(BaseRobustnessPenaltyRewardComputer):
    def __init__(self, weight: float) -> None:
        super().__init__(weight, RobustnessType.I5)

