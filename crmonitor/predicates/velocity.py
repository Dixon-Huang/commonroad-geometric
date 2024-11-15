from enum import Enum
import logging
import math
from typing import List

import numpy as np

from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates.position import PredInFrontOf, PredInSameLane


logger = logging.getLogger(__name__)


class VelocityPredicates(str, Enum):
    KeepsLaneSpeedLimit = "keeps_lane_speed_limit"
    KeepsTypeSpeedLimit = "keeps_type_speed_limit"
    KeepsFovSpeedLimit = "keeps_fov_speed_limit"
    KeepsBrakeSpeedLimit = "keeps_brake_speed_limit"
    Reverses = "reverses"
    SlowLeadingVehicle = "slow_leading_vehicle"
    PreservesTrafficFlow = "preserves_traffic_flow"
    ExistStandingLeadingVehicle = "exist_standing_leading_vehicle"
    InStandstill = "in_standstill"
    DrivesFaster = "drives_faster"
    DrivesWithSlightlyHigherSpeed = "drives_with_slightly_higher_speed"


class PredGenericSpeedLimit(BasePredicateEvaluator):
    def __init__(self, config: CommentedMap):
        super().__init__(config)

    def get_speed_limit(self, world, time_step, vehicle_ids):
        raise NotImplementedError

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        time_step = time_step
        speed_limit = self.get_speed_limit(world, time_step, vehicle_ids)
        if speed_limit is None:
            rob = math.inf
        else:
            rob = speed_limit + self.eps - vehicle.states_cr[time_step].velocity
        rob = self._scale_speed(rob)
        return rob


class PredLaneSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsLaneSpeedLimit
    arity = 1

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.country = SupportedTrafficSignCountry(config.get("country"))

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        ts_interpreter = TrafficSigInterpreter(
            self.country, world.road_network.lanelet_network
        )
        speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        return speed_limit


class PredTypeSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsTypeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle_type = world.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return self.config["max_interstate_speed_truck"]
        else:
            return None


class PredFovSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsFovSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("fov_speed_limit")


class PredBrSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsBrakeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("braking_speed_limit")


class PredLaneSpeedLimitStar(PredLaneSpeedLimit):
    predicate_name = "keeps_lane_speed_limit_star"
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        speed_limit = super(PredLaneSpeedLimitStar, self).get_speed_limit(
            world, time_step, vehicle_ids
        )
        if speed_limit is None:
            speed_limit = self.config["desired_interstate_velocity"]
        return speed_limit


class PredReverses(BasePredicateEvaluator):
    """
    Evaluates if a vehicle drives backwards
    """

    predicate_name = VelocityPredicates.Reverses
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        if vehicle.get_lon_state(time_step).v < -self.config["standstill_error"]:
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return self._scale_speed(
            -self.config["standstill_error"]
            - vehicle.get_lon_state(time_step).v
            - 1.0e-17
        )


class PredSlowLeadingVehicle(BasePredicateEvaluator):
    """
    Predicate which evaluates if a slow leading vehicle exists if front of a vehicle
    """

    predicate_name = VelocityPredicates.SlowLeadingVehicle
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._lane_speed_limit_evaluator = PredLaneSpeedLimitStar(config)
        self._type_speed_limit_evaluator = PredTypeSpeedLimit(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue
            v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_type = self._type_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_list = [
                vehicle.vehicle_param.get("road_condition_speed_limit"),
                v_max_lane,
                v_type,
            ]
            v_max = min(v for v in v_list if v is not None)
            if (
                v_max - veh_o.get_lon_state(time_step).v
                >= self.config["min_velocity_dif"]
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        rob_slow_leading_list = [self._scale_speed(-np.inf)]
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            # Change: Added try-except block to handle exceptions
            try:
                if veh_o.get_lon_state(time_step) is None:
                    continue
                if not self._in_front_of_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                ) or not self._same_lane_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                ):
                    continue
            except Exception as e:
                # logger.debug(f"Error in slow leading vehicle: {e}")
                continue

            v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_type = self._type_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_list = [
                vehicle.vehicle_param.get("road_condition_speed_limit"),
                v_max_lane,
                v_type,
            ]
            v_max = min(v for v in v_list if v is not None)
            rob_slow_leading_list.append(
                self._scale_speed(
                    v_max
                    - veh_o.get_lon_state(time_step).v
                    - self.config["min_velocity_dif"]
                    - 1.0e-17
                )
            )
        return max(rob_slow_leading_list)


class PredPreservesTrafficFlow(BasePredicateEvaluator):
    """
    Predicate for minimum speed limit evaluation
    """

    predicate_name = VelocityPredicates.PreservesTrafficFlow
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._lane_speed_limit_evaluator = PredLaneSpeedLimitStar(config)
        self._type_speed_limit_evaluator = PredTypeSpeedLimit(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_type = self._type_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_list = [
            vehicle.vehicle_param.get("road_condition_speed_limit"),
            vehicle.vehicle_param.get("fov_speed_limit"),
            vehicle.vehicle_param.get("braking_speed_limit"),
            v_max_lane,
            v_type,
        ]
        v_max = min(v for v in v_list if v is not None)
        if v_max - vehicle.get_lon_state(time_step).v < self.config["min_velocity_dif"]:
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_type = self._type_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_list = [
            vehicle.vehicle_param.get("road_condition_speed_limit"),
            vehicle.vehicle_param.get("fov_speed_limit"),
            vehicle.vehicle_param.get("braking_speed_limit"),
            v_max_lane,
            v_type,
        ]
        v_max = min(v for v in v_list if v is not None)
        return self._scale_speed(
            self.config["min_velocity_dif"]
            - v_max
            + vehicle.get_lon_state(time_step).v
            - 1.0e-17
        )


class PredInStandStill(BasePredicateEvaluator):
    """
    Evaluation if vehicle is standing
    """

    predicate_name = VelocityPredicates.InStandstill
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])

        if (
            -self.config["standstill_error"]
            < vehicle.get_lon_state(time_step).v
            < self.config["standstill_error"]
        ):
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return self._scale_speed(
            min(
                vehicle.get_lon_state(time_step).v + self.config["standstill_error"],
                self.config["standstill_error"]
                - vehicle.get_lon_state(time_step).v
                - 1.0e-17,
            )
        )


class PredExistStandingLeadingVehicle(BasePredicateEvaluator):
    """
    Predicate which checks if a standing leading vehicle exist in front of a vehicle
    """

    predicate_name = VelocityPredicates.ExistStandingLeadingVehicle
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._in_standstill_evaluator = PredInStandStill(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue
            if self._in_standstill_evaluator.evaluate_boolean(
                world, time_step, [veh_o.id]
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        rob_standstill_list = [self._scale_speed(-np.inf)]
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue

            rob_standstill_list.append(
                self._in_standstill_evaluator.evaluate_robustness(
                    world, time_step, [veh_o.id]
                )
            )
        return max(rob_standstill_list)


class PredDrivesFaster(BasePredicateEvaluator):
    """
    Predicate which checks if the kth vehicle drives faster than the pth vehicle
    """

    predicate_name = VelocityPredicates.DrivesFaster
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_speed(
            vehicle_k.get_lon_state(time_step).v
            - vehicle_p.get_lon_state(time_step).v
            - 1.0e-17
        )


class PredDrivesWithSlightlyHigherSpeed(BasePredicateEvaluator):
    """
    Predicate which checks if the kth vehicle drives maximum with slightly higher speed than the pth vehicle
    """

    predicate_name = VelocityPredicates.DrivesWithSlightlyHigherSpeed
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_speed(
            min(
                vehicle_k.get_lon_state(time_step).v
                - vehicle_p.get_lon_state(time_step).v
                - 1.0e-17,
                self.config["slightly_higher_speed_difference"]
                - vehicle_k.get_lon_state(time_step).v
                + vehicle_p.get_lon_state(time_step).v
                - 1.0e-17,
            )
        )
