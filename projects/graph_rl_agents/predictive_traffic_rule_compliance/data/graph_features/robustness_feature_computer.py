import warnings
from typing import Optional, Dict, Union

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State
from torch import Tensor
import logging

from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.types import (
    V_Feature,
    V2V_Feature,
)
from commonroad_geometric.dataset.extraction.traffic.feature_computers import (
    BaseFeatureComputer,
    VFeatureParams,
    FeatureDict,
    V2VFeatureParams,
    T_FeatureParams,
)
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (
    ScenarioSimulation,
)
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.common.vehicle import ControlledVehicle
from enum import Enum
from typing import Dict, List, Callable


logger = logging.getLogger(__name__)


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


class RuleRobustnessFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    def __init__(self, active_robustness: List[RobustnessType] = [RobustnessType.G1]) -> None:
        super().__init__()
        self._rule_monitors: Dict[int, Dict[RobustnessType, RuleEvaluator]] = {}
        self._robustness_values = {}
        self.active_robustness = active_robustness or list(RobustnessType)

    def __call__(
            self,
            params: VFeatureParams,
            simulation: ScenarioSimulation,
    ) -> FeatureDict:
        if not params.is_ego_vehicle:
            return {}

        if self._is_vehicle_offroad(simulation):
            return self._get_nan_features()

        features = {}
        for robustness_type in self.active_robustness:
            evaluator = self._get_evaluator(simulation, params.obstacle, robustness_type.value)
            robustness_series = evaluator.evaluate()
            self._robustness_values.setdefault(params.obstacle.obstacle_id, {})[robustness_type] = robustness_series
            features[f"{robustness_type.name}Robustness"] = robustness_series[-1]

        return features

    def _is_vehicle_offroad(self, simulation: ScenarioSimulation) -> bool:
        return len(
            simulation.current_obstacles[-1]._prediction._shape_lanelet_assignment[simulation.current_time_step]) <= 0

    def _get_nan_features(self) -> Dict[str, float]:
        return {f"{r.name}Robustness": np.nan for r in self.active_robustness}

    def _get_evaluator(
            self,
            simulation: ScenarioSimulation,
            obstacle: DynamicObstacle,
            rule: str,
    ) -> RuleEvaluator:
        veh = simulation.world_state.vehicle_by_id(obstacle.obstacle_id)
        assert veh is not None, f"The obstacle {obstacle.obstacle_id} is not converted to a vehicle in the world state!"

        if obstacle.obstacle_id not in self._rule_monitors:
            self._rule_monitors[obstacle.obstacle_id] = {}

        if rule not in self._rule_monitors[obstacle.obstacle_id]:
            self._rule_monitors[obstacle.obstacle_id][rule] = RuleEvaluator.create_from_config(simulation.world_state,
                                                                                               veh, rule)

        return self._rule_monitors[obstacle.obstacle_id][rule]

    def reset(self, simulation: ScenarioSimulation) -> None:
        self._rule_monitors.clear()
        self._robustness_values.clear()

    def set_active_robustness(self, robustness_types: List[RobustnessType]) -> None:
        self.active_robustness = robustness_types


class PredicateRobustnessFeatureComputer(BaseFeatureComputer[V2VFeatureParams]):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._feature_keys = None

    def __call__(
        self,
        params: V2VFeatureParams,
        simulation: ScenarioSimulation,
    ) -> FeatureDict:
        src = params.source_obstacle
        dst = params.target_obstacle

        if src.obstacle_id == dst.obstacle_id:
            return {}

        ws = simulation.world_state

        src_veh = ws.vehicle_by_id(src.obstacle_id)
        predicates = src_veh.predicate_cache[params.time_step, :, (dst.obstacle_id,)]
        if not self._feature_keys:
            self._feature_keys = set(predicates.keys())

        return {k: predicates.setdefault(k, np.nan) for k in self._feature_keys}

    def _setup(
        self,
        params: T_FeatureParams,
        simulation: ScenarioSimulation,
    ) -> None:
        for vehicle in simulation.world_state.vehicles:
            evaluator = RuleEvaluator.create_from_config(
                simulation.world_state, vehicle
            )
            evaluator.evaluate()
