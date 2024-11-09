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


logger = logging.getLogger(__name__)


class RuleRobustnessFeatureComputer(BaseFeatureComputer[VFeatureParams]):
    def __init__(self) -> None:
        super().__init__()
        self._rule_monitors: Optional[Dict[int, RuleEvaluator]] = None
        self._robustness_values = {}

    def __call__(
        self,
        params: VFeatureParams,
        simulation: ScenarioSimulation,
    ) -> FeatureDict:
        # hacky way to deal with Ego Vehicle offroad problem
        if (
            len(
                simulation.current_obstacles[-1]._prediction._shape_lanelet_assignment[
                    simulation.current_time_step
                ]
            )
            <= 0
        ):
            logger.warning(
                f"Ego vehicle at time step {simulation.current_time_step} has no lanelet assignment due to off-road, return the robustness feature of vehicle {params.obstacle.obstacle_id} with nan!"
            )
            robustness_g1 = np.nan
            robustness_g2 = robustness_g3 = np.nan
            features = {
                V_Feature.G1Robustness.value: robustness_g1,
                V_Feature.G2Robustness.value: robustness_g2,
                V_Feature.G3Robustness.value: robustness_g3,
            }
            return features

        evaluator_g1 = self._get_evaluator(simulation, params.obstacle, "R_G1")
        evaluator_g2 = self._get_evaluator(simulation, params.obstacle, "R_G2")
        evaluator_g3 = self._get_evaluator(simulation, params.obstacle, "R_G3")
        robustness_series_g1 = evaluator_g1.evaluate()
        robustness_series_g2 = evaluator_g2.evaluate()
        robustness_series_g3 = evaluator_g3.evaluate()
        self._robustness_values[params.obstacle.obstacle_id] = (
            robustness_series_g1,
            robustness_series_g2,
            robustness_series_g3,
        )

        robustness_g1 = robustness_series_g1[-1]
        robustness_g2 = robustness_series_g2[-1]
        robustness_g3 = robustness_series_g3[-1]

        features = {
            V_Feature.G1Robustness.value: robustness_g1,
            V_Feature.G2Robustness.value: robustness_g2,
            V_Feature.G3Robustness.value: robustness_g3,
        }

        return features

    def _get_evaluator(
        self,
        simulation: ScenarioSimulation,
        obstacle: DynamicObstacle,
        rule: str = "R_G1",
    ) -> RuleEvaluator:
        veh = simulation.world_state.vehicle_by_id(obstacle.obstacle_id)
        assert (
            veh is not None
        ), f"The obstacle {obstacle.obstacle_id} is not converted to a vehicle in the world state!"
        if isinstance(veh, ControlledVehicle):
            self._rule_monitors[
                obstacle.obstacle_id
            ] = RuleEvaluator.create_from_config(simulation.world_state, veh, rule)
            return self._rule_monitors[obstacle.obstacle_id]
        if obstacle.obstacle_id not in self._rule_monitors:
            self._rule_monitors[
                obstacle.obstacle_id
            ] = RuleEvaluator.create_from_config(simulation.world_state, veh, rule)
        return self._rule_monitors[obstacle.obstacle_id]

    def reset(
        self,
        simulation: ScenarioSimulation,
    ) -> None:
        # The reset method is called at the beginning of a new scenario.
        self._rule_monitors = {}
        self._robustness_values.clear()


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
