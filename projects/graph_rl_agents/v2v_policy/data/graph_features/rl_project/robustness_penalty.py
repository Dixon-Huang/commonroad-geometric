import numpy as np

from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import (
    BaseRewardComputer,
)
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import (
    MissingFeatureException,
)
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import (
    EgoVehicleSimulation,
)
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import (
    V_Feature,
)


class G1RobustnessPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
    ) -> None:
        self._weight = weight
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
    ) -> float:
        try:
            robustness_g1 = float(
                data.v[V_Feature.G1Robustness.value][-1][0].nan_to_num_(nan=-1)
            )
        except KeyError:
            raise MissingFeatureException("g1_robustness")

        # loss = np.array([1 - robustness_g1, 1 - robustness_g2, 1 - robustness_g3])
        # weight = np.array(self._weight)
        penalty = robustness_g1 * self._weight
        return penalty


class G2RobustnessPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
    ) -> None:
        self._weight = weight
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
    ) -> float:
        try:
            robustness_g2 = float(
                data.v[V_Feature.G2Robustness.value][-1][0].nan_to_num_(nan=-1)
            )
        except KeyError:
            raise MissingFeatureException("g2_robustness")

        penalty = robustness_g2 * self._weight
        return penalty


class G3RobustnessPenaltyRewardComputer(BaseRewardComputer):
    def __init__(
        self,
        weight: float,
    ) -> None:
        self._weight = weight
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
    ) -> float:
        try:
            robustness_g3 = float(
                data.v[V_Feature.G3Robustness.value][-1][0].nan_to_num_(nan=-1)
            )
        except KeyError:
            raise MissingFeatureException("g3_robustness")

        penalty = robustness_g3 * self._weight
        return penalty
