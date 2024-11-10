import enum
import math
from decimal import Decimal
from functools import reduce
from pathlib import Path
from typing import Dict, Union, List, Tuple, Iterable, Sequence

import numba
import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.trajectory import State
from ruamel.yaml import YAML
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from crmonitor.common.road_network import RoadNetwork, Lane
from crmonitor.common.vehicle import Vehicle


@enum.unique
class OperatingMode(enum.Enum):
    MONITOR = "monitor"
    ROBUSTNESS = "robustness"

def create_ego_vehicle_param(ego_vehicle_param: Dict, dt: float) -> Dict:
    """
    Update ego vehicle parameters

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :returns updated dictionary with parameters of ACC vehicle
    """
    if ego_vehicle_param.get("vehicle_number") == 1:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif ego_vehicle_param.get("vehicle_number") == 2:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif ego_vehicle_param.get("vehicle_number") == 3:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError(
            "Wrong vehicle number for ACC vehicle in config file defined.")

    emergency_profile = ego_vehicle_param.get("emergency_profile")
    emergency_profile += [ego_vehicle_param.get(
        "j_min")] * ego_vehicle_param.get("emergency_profile_num_steps_fb")
    ego_vehicle_param["emergency_profile"] = emergency_profile

    if (not -1e-12 <= (Decimal(str(ego_vehicle_param.get("t_react"))) % Decimal(
            str(dt))) <= 1e-12):
        raise ValueError("Reaction time must be multiple of time step size.")

    return ego_vehicle_param


def create_other_vehicles_param(other_vehicles_param: Dict) -> Dict:
    """
    Update other vehicle's parameters

    :param other_vehicles_param: dictionary with physical parameters of other vehicles
    :returns updated dictionary with parameters of other vehicles
    """
    if other_vehicles_param.get("vehicle_number") == 1:
        other_vehicles_param["dynamics_param"] = parameters_vehicle1()
    elif other_vehicles_param.get("vehicle_number") == 2:
        other_vehicles_param["dynamics_param"] = parameters_vehicle2()
    elif other_vehicles_param.get("vehicle_number") == 3:
        other_vehicles_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError(
                "Wrong vehicle number for leading vehicle in config file defined.")

    return other_vehicles_param


def create_simulation_param(simulation_param: Dict, dt: float,
                            country: str) -> Dict:
    """
    Update simulation parameters

    :param simulation_param: dictionary with parameters of the simulation environment
    :param country: country of CommonRoad scenario
    :param dt: time step size of CommonRoad scenario
    :returns updated dictionary with parameters of CommonRoad scenario
    """
    simulation_param["dt"] = dt
    simulation_param["country"] = SupportedTrafficSignCountry(country)

    return simulation_param


def calc_v_max_fov(ego_vehicle_param: Dict, simulation_param: Dict) -> int:
    """
    Calculates safety (field of view) based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :returns maximum allowed velocity
    """
    v_ego = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    a_min = ego_vehicle_param.get("a_min") + ego_vehicle_param.get("a_corr")
    emergency_profile = ego_vehicle_param.get("emergency_profile")
    (a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego, stopping_distance,
     t_react, v_max, v_min,) = init_v_max_calculation(a_min, ego_vehicle_param,
            emergency_profile, simulation_param, v_ego)
    while (dist_offset <= 0 or dist_offset >= 0.5 and not (
            v_max == ego_vehicle_param.get(
            "dynamics_param").longitudinal.v_max and dist_offset > 0.5)):
        if (ego_vehicle_param.get(
                "fov") - stopping_distance - ego_vehicle_param.get(
                "const_dist_offset") < 0):
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > ego_vehicle_param.get("dynamics_param").longitudinal.v_max:
            v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt,
                t_react, a_min, a_max, j_max, v_min, v_max, a_corr,
                emergency_profile, )
        dist_offset = (ego_vehicle_param.get(
            "fov") - stopping_distance - ego_vehicle_param.get(
            "const_dist_offset"))

    return math.floor(v_max)


def calc_v_max_braking(ego_vehicle_param: Dict, simulation_param: Dict,
        traffic_rule_param: Dict) -> int:
    """
    Calculates braking based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param traffic_rule_param: dictionary with parameters related to traffic rules
    :returns maximum allowed velocity
    """
    v_max_delta = ego_vehicle_param.get(
            "dynamics_param").longitudinal.v_max - traffic_rule_param.get(
        "max_velocity_limit_free_driving")
    v_ego = v_max_delta
    a_min = traffic_rule_param.get("a_abrupt")
    emergency_profile = 2500 * [traffic_rule_param.get("j_abrupt")]
    (a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego, stopping_distance,
     t_react, v_max, v_min,) = init_v_max_calculation(a_min, ego_vehicle_param,
            emergency_profile, simulation_param, v_ego)
    while (dist_offset <= 0 or dist_offset >= 0.5 and not (
            v_max == v_max_delta and dist_offset > 0.5)):
        if ego_vehicle_param.get("fov") - stopping_distance < 0:
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > v_max_delta:
            v_max = v_max_delta
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt,
                t_react, a_min, a_max, j_max, v_min, v_max, a_corr,
                emergency_profile, )
        dist_offset = (ego_vehicle_param.get(
            "fov") - stopping_distance - ego_vehicle_param.get(
            "const_dist_offset"))

    return math.floor(
        v_max + traffic_rule_param.get("max_velocity_limit_free_driving"))


def init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile,
        simulation_param, v_ego):
    """
    Helper function to initialize values for calculation of maximum velocity based on field of view and braking

    :param a_min: minimum acceleration of the ego vehicle
    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param emergency_profile: emergency jerk profile which is executed in case of a fail-safe braking maneuver
    :param simulation_param: dictionary with parameters of the simulation environment
    :param v_ego: ego vehicle velocity
    :returns different parameters for the calculation of the maximum allowed velocity
    """
    s_ego = 0
    a_ego = 0  # ego vehicle is already at v_max
    dt = simulation_param.get("dt")
    t_react = ego_vehicle_param.get("t_react")
    a_max = ego_vehicle_param.get("a_max")
    a_corr = ego_vehicle_param.get("a_corr")
    j_max = ego_vehicle_param.get("j_max")
    v_min = ego_vehicle_param.get("v_min")
    v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react,
            a_min, a_max, j_max, v_min, v_max, a_corr, emergency_profile, )
    dist_offset = (ego_vehicle_param.get(
        "fov") - stopping_distance - ego_vehicle_param.get("const_dist_offset"))

    return (a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego,
            stopping_distance, t_react, v_max, v_min,)


def emg_stopping_distance(s: float, v: float, a: float, dt: float,
        t_react: float, a_min: float, a_max: float, j_max: float, v_min: float,
        v_max: float, a_corr: float, emergency_profile: List[float], ) -> float:
    """
    Calculates stopping distance of a vehicle which applies predefined emergency jerk profile
     and considering reaction time

    :param s: current longitudinal front position of vehicle
    :param v: current velocity of vehicle
    :param a: current acceleration of vehicle
    :param dt: time step size
    :param t_react: reaction time of vehicle
    :param a_min: minimum acceleration of vehicle
    :param a_max: maximum acceleration of vehicle
    :param j_max: maximum jerk of vehicle
    :param v_max: maximum velocity of vehicle
    :param v_min: minimum velocity of vehicle
    :param a_corr: maximum deviation of vehicle from real acceleration
    :param emergency_profile: jerk emergency profile
    :returns: stopping distance
    """
    # application of reaction time (maximum jerk of vehicle):
    a = min(a + a_corr, a_max)
    if v == v_max:
        a = 0
    steps_reaction_time = round(t_react / dt)
    for i in range(steps_reaction_time):
        s, v, a = vehicle_dynamics_jerk(s, v, a, j_max, v_min, v_max, a_min,
                                        a_max, dt)

    # application of the emergency profile:
    index = 0
    while v > 0:
        a = min(a + a_corr, a_max)
        if v == v_max:
            a = 0
        s, v, a = vehicle_dynamics_jerk(s, v, a, emergency_profile[index],
                v_min, v_max, a_min, a_max, dt)
        index = index + 1

    return s


def vehicle_dynamics_jerk(s_0: float, v_0: float, a_0: float, j_input: float,
        v_min: float, v_max: float, a_min: float, a_max: float, dt: float, ) -> \
Tuple[float, float, float]:
    """
    Applying vehicle dynamics for one times step with jerk as input

    :param s_0: current longitudinal position at vehicle's front
    :param v_0: current velocity of vehicle
    :param a_0: current acceleration of vehicle
    :param j_input: jerk input for vehicle
    :param v_min: minimum velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :param a_min: minimum acceleration of vehicle
    :param a_max: maximum acceleration of vehicle
    :param dt: time step size
    :return: new position, velocity, acceleration
    """
    a_new = a_0 + j_input * dt
    if a_new > a_max:
        t_a = abs((a_max - a_0) / j_input)  # time until a_max is reached
        a_new = a_max
    elif a_new < a_min:
        t_a = abs((a_0 - a_min) / j_input)  # time until a_min is reached
        a_new = a_min
    else:
        t_a = dt

    v_new = v_0 + a_0 * dt + 0.5 * j_input * t_a ** 2
    if v_new > v_max and j_input != 0.0:
        t_v = calculate_tv(a_0, j_input, v_0,
                           v_max)  # time until v_max is reached
        t_a = t_v
        v_new = v_max
    elif v_new > v_max and j_input == 0.0:
        t_v = abs((v_max - v_0) / a_0)
        t_a = t_v
        v_new = v_max
    if v_new < v_min and j_input != 0.0:
        t_v = calculate_tv(a_0, j_input, v_0,
                           v_min)  # time until v_min is reached
        t_a = t_v
        v_new = v_min
    elif v_new < v_min and j_input == 0.0:
        t_v = abs((v_0 - v_min) / a_0)
        t_a = t_v
        v_new = v_min
    else:
        t_v = dt

    if v_new == v_max or v_new == v_min:
        a_new = 0

    s_new = s_0 + v_0 * t_v + 0.5 * a_0 * t_a ** 2 + (
                1 / 6) * j_input * t_a ** 3

    return s_new, v_new, a_new


def calculate_tv(a_0, j_input, v_0, v_max):
    """
    Calculates time how long input can be applied until minimum/maximum velocity is reached

    :param a_0: current acceleration of vehicle
    :param j_input: jerk input for vehicle
    :param v_0: current velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :returns time until v_max is reached
    """
    d = abs(a_0) ** 2 - 4 * 0.5 * abs(j_input) * (v_max - v_0)
    t_1 = (-abs(a_0) + math.sqrt(d)) / (2 * 0.5 * abs(j_input))
    t_2 = (-abs(a_0) - math.sqrt(d)) / (2 * 0.5 * abs(j_input))
    t_v = min(abs(t_1), abs(t_2))

    return t_v

def get_robust_lanelet_assignment(state: State, obs: DynamicObstacle, road_network: RoadNetwork):
    lanelets = obs.initial_shape_lanelet_ids
    lanes = road_network.find_lanes_by_lanelets(lanelets)
    shape = obs.obstacle_shape.rotate_translate_local(state.position,
                                                      state.orientation)
    veh_area = shape.shapely_object.area
    intersecting_lanes = set()
    for lane in lanes:
        intersection = shape.shapely_object.intersection(
            lane.lanelet.convert_to_polygon().shapely_object)
        intersect_area = intersection.area
        if intersect_area / veh_area > 0.33:
            intersecting_lanes.add(lane.lanelet.lanelet_id)
    return intersecting_lanes


def create_vehicle(obstacle: DynamicObstacle, vehicle_param: Dict,
        road_network: RoadNetwork, dt: float,
        ego_vehicle: Vehicle = None, create_robust_lanelet_assignment=False) -> Vehicle:
    """
    Transforms a CommonRoad obstacle to a vehicle object

    :param obstacle: CommonRoad obstacle
    :param vehicle_param: dictionary with vehicle parameters
    :param ego_vehicle: ego vehicle object (if it exist already) for reference generation
    :param road_network: CommonRoad lanelet network
    :param dt: time step size
    :return: vehicle object
    """
    acceleration = _compute_acceleration(obstacle.initial_state.velocity,
            obstacle.prediction.trajectory.state_list[0].velocity, dt, )
    jerk = _compute_jerk(acceleration, 0, dt)
    if ego_vehicle is None:
        initial_lanelets = [
                road_network.lanelet_network.find_lanelet_by_id(lanelet_id) for
                lanelet_id in obstacle.initial_shape_lanelet_ids]
        if LaneletType.ACCESS_RAMP in initial_lanelets[0].lanelet_type:
            main_carriage_way_lanelet_id = _find_main_carriage_way_lanelet_id(
                    initial_lanelets[0], road_network)
            lane = road_network.find_lane_by_obstacle(
                    [main_carriage_way_lanelet_id], [])
        else:
            lane = road_network.find_lane_by_obstacle(
                    list(obstacle.initial_center_lanelet_ids),
                    list(obstacle.initial_shape_lanelet_ids), )
        reference_lane = lane
    else:
        lane = road_network.find_lane_by_obstacle(
            list(obstacle.initial_center_lanelet_ids),
            list(obstacle.initial_shape_lanelet_ids))
        reference_lane = ego_vehicle.lane

    state_lon, state_lat = create_curvilinear_states(
            obstacle.initial_state.position, obstacle.initial_state.velocity,
            acceleration, jerk, obstacle.initial_state.orientation,
            reference_lane, )

    initial_time_step = obstacle.initial_state.time_step
    state_list_lon = {initial_time_step: state_lon}
    state_list_lat = {initial_time_step: state_lat}
    state_list_cr = {initial_time_step: obstacle.initial_state}
    signal_series = {initial_time_step: obstacle.initial_signal_state}
    lanelet_assignments = {
            initial_time_step: obstacle.initial_shape_lanelet_ids}
    if create_robust_lanelet_assignment:
        robust_lanelet_assginment = {initial_time_step: get_robust_lanelet_assignment(obstacle.initial_state, obstacle, road_network)}
    else:
        robust_lanelet_assignment = None
    for state in obstacle.prediction.trajectory.state_list:
        acceleration = _compute_acceleration(state_lon.v, state.velocity, dt)
        if state.time_step - 1 in state_list_lon:
            previous_acceleration = state_list_lon[state.time_step - 1].a
        else:  # previous state out of projection domain
            previous_acceleration = 0.0
        jerk = _compute_jerk(acceleration, previous_acceleration,
                dt)  # TODO: why calculating jerk without previous acceleration?

        state_lon, state_lat = create_curvilinear_states(state.position,
                state.velocity, acceleration, jerk, state.orientation,
                reference_lane, )
        if state_lon is None or state_lat is None:
            break
        state_list_lon[state.time_step] = state_lon
        state_list_lat[state.time_step] = state_lat
        state_list_cr[state.time_step] = state
        signal_series[state.time_step] = obstacle.signal_state_at_time_step(
                state.time_step)
        lanelet_assignments[state.time_step] = \
        obstacle.prediction.shape_lanelet_assignment[state.time_step]
        if create_robust_lanelet_assignment:
            robust_lanelet_assginment[state.time_step] = get_robust_lanelet_assignment(state, obstacle, road_network)

    vehicle = Vehicle(state_list_lon, state_list_lat, obstacle.obstacle_shape, state_list_cr, obstacle.obstacle_id,
                      obstacle.obstacle_type, vehicle_param, lanelet_assignments, signal_series, lane,
                      robust_lanelet_assignment)
    return vehicle


def _adjacent_to_ego(ego_lanelet_id: int, obs_lanelet_id: int,
        road_network: RoadNetwork) -> bool:
    """
    Evaluates if a vehicle is in a to the ego vehicle adjacent lane

    :param ego_lanelet_id: IDs of lanelets the ego vehicle is on
    :param obs_lanelet_id: IDs of lanelets the other vehicle is on
    :param road_network: CommonRoad lanelet network
    :return: boolean indicating if the vehicle is in an adjacent lane
    """
    adjacent_lanelet_ids = {ego_lanelet_id}
    ego_lanelet = road_network.lanelet_network.find_lanelet_by_id(
        ego_lanelet_id)
    current_lanelet = ego_lanelet
    while (
            current_lanelet.adj_left_same_direction is not None and current_lanelet.adj_left_same_direction is True):
        current_lanelet = road_network.lanelet_network.find_lanelet_by_id(
                current_lanelet.adj_left)
        adjacent_lanelet_ids.add(current_lanelet.lanelet_id)
    while (
            current_lanelet.adj_right_same_direction is not None and current_lanelet.adj_right_same_direction is True):
        current_lanelet = road_network.lanelet_network.find_lanelet_by_id(
                current_lanelet.adj_right)
        adjacent_lanelet_ids.add(current_lanelet.lanelet_id)
    for lanelet_id in list(adjacent_lanelet_ids):
        lane = road_network.find_lane_by_lanelet(lanelet_id)
        if obs_lanelet_id in lane.contained_lanelets:
            return True
    return False


def _find_main_carriage_way_lanelet_id(lanelet: Lanelet, road_network) -> int:
    """
    Searches for an adjacent lanelet part of the main carriageway

    :param lanelet: start lanelet
    :return: ID of a lanelet which is part of the main carriageway
    """
    current_lanelet = lanelet
    if LaneletType.MAIN_CARRIAGE_WAY in current_lanelet.lanelet_type:
        return current_lanelet.lanelet_id
    while current_lanelet.adj_left_same_direction is not None:
        current_lanelet = road_network.lanelet_network.find_lanelet_by_id(
                current_lanelet.adj_left)
        if LaneletType.MAIN_CARRIAGE_WAY in current_lanelet.lanelet_type:
            return current_lanelet.lanelet_id


def _compute_jerk(current_acceleration: float, previous_acceleration: float,
        dt: float) -> float:
    """
    Computes jerk given acceleration

    :param current_acceleration: acceleration of current time step
    :param previous_acceleration: acceleration of previous time step
    :param dt: time step size
    :return: jerk
    """
    jerk = (current_acceleration - previous_acceleration) / dt
    return jerk


def _compute_acceleration(previous_velocity: float, current_velocity: float,
                          dt: float):
    """
    Computes acceleration given velocity

    :param current_velocity: velocity of current time step
    :param previous_velocity: velocity of previous time step
    :param dt: time step size
    :return: acceleration
    """
    acceleration = (current_velocity - previous_velocity) / dt
    return acceleration


def create_scenario_vehicles(dt: float, ego_obstacle: DynamicObstacle,
        ego_vehicle_param: Dict, other_vehicles_param: Dict,
        road_network: RoadNetwork,
        dynamic_obstacles: List[DynamicObstacle],
        obstacle_vehicle_curvi_states_dict: Union[Dict, None] = None) -> Tuple[Vehicle, List[Vehicle]]:
    """
    Creates vehicles object for all obstacles within a CommonRoad scenario given

    :param obstacle_vehicle_curvi_states_dict:
    :param ego_obstacle: CommonRoad obstacle of ego vehicle
    :param dt: time step size
    :param ego_vehicle_param: :param vehicle_param: dictionary with vehicle ego parameters
    :param other_vehicles_param: :param vehicle_param: dictionary with vehicle parameters of other traffic participants
    :param road_network: road network
    :param dynamic_obstacles: list containing dynamic obstacles of CommonRoad scenario

    :return: ego vehicle object and list of vehicle objects containing other traffic participants
    """
    other_vehicles = []
    ego_vehicle = create_vehicle(ego_obstacle, ego_vehicle_param, road_network, dt)
    for obs in dynamic_obstacles:
        if (
                obs.obstacle_id == ego_obstacle.obstacle_id or obs.prediction is None or obs.initial_state.time_step >
                ego_obstacle.prediction.trajectory.state_list[
                    -1].time_step or ego_obstacle.initial_state.time_step >
                obs.prediction.trajectory.state_list[-1].time_step):
            continue
        if obstacle_vehicle_curvi_states_dict is None:
            vehicle = create_vehicle(obs, other_vehicles_param, road_network, dt,
                                     ego_vehicle)
        else:
            # load curvi states
            state_list_lon, state_list_lat = obstacle_vehicle_curvi_states_dict[obs.obstacle_id][
                ego_vehicle.lane.lanelet.lanelet_id]

            initial_time_step = obs.initial_state.time_step
            state_list_cr = {initial_time_step: obs.initial_state}
            lanelet_assignments = {initial_time_step: obs.initial_shape_lanelet_ids}
            signal_series = {initial_time_step: obs.initial_signal_state}
            for state in obs.prediction.trajectory.state_list:
                state_list_cr[state.time_step] = state
                signal_series[state.time_step] = obs.signal_state_at_time_step(state.time_step)
                lanelet_assignments[state.time_step] = obs.prediction.shape_lanelet_assignment[state.time_step]
            lane = road_network.find_lane_by_obstacle(list(obs.initial_center_lanelet_ids),
                                                      list(obs.initial_shape_lanelet_ids))

            vehicle = Vehicle(state_list_lon, state_list_lat, obs.obstacle_shape, state_list_cr, obs.obstacle_id,
                              obs.obstacle_type, other_vehicles_param, lanelet_assignments, signal_series, lane, None)

        other_vehicles.append(vehicle)
    return ego_vehicle, other_vehicles


def load_yaml(file_name: Union[Path, str]) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    file_name = Path(file_name)
    config = YAML().load(file_name)
    return config


def update_vehicle(obstacle: DynamicObstacle, dt: float, time_step: int,
        reference_lane: Lane, vehicle: Vehicle, ) -> Vehicle:
    """
    Append state to Vehicle according to current CommonRoad Obstacle State
    :param obstacle: CommonRoad Obstacle which contains TrajectoryPrediction
    :param dt: time step size of scenario
    :param time_step: current time step
    :param reference_lane: reference Lane object of Ego Vehicle
    :param vehicle: the Vehicle object to be updated
    :return: updated Vehicle object
    """
    # get obstacle current state
    obstacle_state = obstacle.state_at_time(time_step)

    if time_step - 1 in vehicle.states_lon:
        previous_state_longitudinal = vehicle.states_lon[time_step - 1]
        previous_v = previous_state_longitudinal.v
        previous_a = previous_state_longitudinal.a
    # initial state
    elif time_step - 1 == obstacle.initial_state.time_step:
        previous_v = obstacle.initial_state.velocity
        if hasattr(obstacle.initial_state, "acceleration"):
            previous_a = obstacle.initial_state.acceleration
        else:
            previous_a = 0.0
    # out of projection domain
    else:
        previous_state = obstacle.state_at_time(time_step - 1)
        previous_v = previous_state.velocity
        if hasattr(previous_state, "acceleration"):
            previous_a = previous_state.acceleration
        else:
            previous_a = 0.0

    # get acceleration or compute from previous and current velocity
    if hasattr(obstacle_state, "acceleration"):
        acceleration = obstacle_state.acceleration
    else:
        acceleration = _compute_acceleration(previous_v,
                                             obstacle_state.velocity, dt)

    # compute jerk from current and previous acceleration
    jerk = _compute_jerk(acceleration, previous_a, dt)
    state_lon, state_lat = create_curvilinear_states(obstacle_state.position,
            obstacle_state.velocity, acceleration, jerk,
            obstacle_state.orientation, reference_lane, )
    if state_lon is None or state_lat is None:  # out of projection
        return vehicle
    else:
        lanelet_assignment = obstacle.prediction.shape_lanelet_assignment[time_step]
        vehicle.append_time_step(time_step, state_lon, state_lat,
                obstacle_state, lanelet_assignment, signal_state=None, )
        return vehicle


def update_scenario_vehicles(dt: float, time_step: int,
        ego_obstacle: DynamicObstacle, dynamic_obstacles: List[DynamicObstacle],
        ego_vehicle: Vehicle, dynamic_vehicles: Dict[int, Vehicle],
        other_vehicles_param: Dict, road_network: RoadNetwork,
        obstacle_vehicle_dict: Union[Dict, None] = None, ) -> Tuple[Vehicle, List[Vehicle]]:
    """
    Append states for all Vehicles according to CommmonRoad Obstacle States
    :param dt: time step size
    :param time_step: current time step
    :param ego_obstacle: CommonRoad DynamicObstacle for ego vehicle
    :param dynamic_obstacles: List of all CommonRoad DynamicObstacles
    :param ego_vehicle: ego Vehicle to be updated
    :param dynamic_vehicles: dynamic Vehicles to be updated
    :param other_vehicle_param: dictionary with vehicle parameters
    :param road_network: CommonRoad lanelet network
    :return: updated ego and dynamic vehicles
    """
    # update ego vehicle # TODO: ego_vehicle.lane is not updated
    ego_vehicle = update_vehicle(ego_obstacle, dt, time_step, ego_vehicle.lane, ego_vehicle)

    # update obstacle vehicles
    other_vehicles = []
    for o in dynamic_obstacles:
        # only update if obstacle appears at the current time step
        if (o.initial_state.time_step <= time_step <= o.prediction.trajectory.final_state.time_step):
            if obstacle_vehicle_dict is None:
                if o.obstacle_id in dynamic_vehicles:
                    updated_vehicle = update_vehicle(o, dt, time_step,
                            ego_vehicle.lane, dynamic_vehicles[o.obstacle_id])
                else:
                    # vehicle was not created yet, create new vehicle
                    updated_vehicle = create_vehicle(o, other_vehicles_param, road_network, dt, ego_vehicle)
            else:
                if o.obstacle_id not in dynamic_vehicles:
                    vehicle = obstacle_vehicle_dict[o.obstacle_id][ego_vehicle.lane.lanelet.lanelet_id]
                    # update vehicle classification and lane
                    vehicle.lane = road_network.find_lane_by_obstacle(list(o.initial_center_lanelet_ids),
                                                                      list(o.initial_shape_lanelet_ids))

            other_vehicles.append(updated_vehicle)

    return ego_vehicle, other_vehicles


def return_true_or_inf(operating_mode: OperatingMode) -> Union[float, bool]:
    """
    Helper function to return True or inf for robustness mode
    :param operating_mode:
    :return:
    """
    return math.inf if operating_mode is OperatingMode.ROBUSTNESS else True


def return_false_or_minf(operating_mode: OperatingMode) -> Union[float, bool]:
    """
    Helper function to return True or inf for robustness mode
    :param operating_mode:
    :return:
    """
    return -math.inf if operating_mode is OperatingMode.ROBUSTNESS else False


def gather(l: Sequence, indices: Iterable[int]) -> Tuple:
    output = []
    for i in indices:
        output.append(l[i])
    return tuple(output)


def flatten_nested_dict(data, path=tuple()):
    entries = []
    for key, val in data.items():
        if isinstance(val, dict):
            entries.extend(flatten_nested_dict(val, path + (key,)))
        else:
            entries.append(path + (key, val))
    return entries


@numba.njit(fastmath=True)
def min_max(arr):
    """
    https://stackoverflow.com/questions/12200580/numpy-function-for-simultaneous-max-and-min
    :param arr:
    :return:
    """
    n = arr.size
    odd = n % 2
    if not odd:
        n -= 1
    max_val = min_val = arr[0]
    i = 1
    while i < n:
        x = arr[i]
        y = arr[i + 1]
        if x > y:
            x, y = y, x
        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2
    if not odd:
        x = arr[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)
    return min_val, max_val


def union_set(s: Iterable):
    return reduce(lambda agg, e: agg.union(e), s, set())


def cartesian_to_curvilinear(reference_paths: Iterable[np.ndarray], cartesian_points: np.ndarray) -> np.ndarray:
    curvilinear_coords = []
    for ref_path in reference_paths:
        # This could also be pre-computed
        # Segment start points
        seg_start = ref_path[:-1]
        # Segment directions
        seg_dir = ref_path[1:] - ref_path[:-1]
        # Segment lengths
        seg_length = np.sqrt(np.sum(seg_dir * seg_dir, axis=-1))
        # Cumulated segment length offsets
        cumsum_seg_length = np.concatenate((np.zeros(1), np.cumsum(seg_length)))
        # Segment direction with length 1
        seg_dir_normalized = seg_dir / seg_length[:, np.newaxis]
        # Segment normal with length 1
        seg_normal_normalized = seg_dir_normalized[:, ::-1].copy()
        seg_normal_normalized[:, 1] *= -1

        # Calculate the normalized segment arc length value
        s_norm_segment = ((cartesian_points[:, np.newaxis] - seg_start[np.newaxis]) * seg_dir_normalized).sum(axis=-1)
        # Calculate the arc length relative to the segment length
        s = s_norm_segment / seg_length[np.newaxis, :]
        # Determine the segment with the lowest index to which the points can be projected (0 <= s <= 1)
        # np.argmax of a boolean gets the index of the first True value, or 0 if there is no true value.
        idx = np.argmax((0 <= s) & (s <= 1), axis=-1)
        arc_length = s_norm_segment[np.arange(start=0, stop=len(idx)), idx]
        # Get the projected point on the line
        projected_points = ref_path[idx] + seg_dir_normalized[idx] * arc_length[:, np.newaxis]
        # Get the signed lateral distance
        lateral_dist = -((cartesian_points - projected_points) * seg_normal_normalized[idx]).sum(axis=-1)
        # Offset arc lengths by the arc length of the segment start
        arc_length = arc_length + cumsum_seg_length[idx]
        # Assemble curvilinear coordinate array
        cc = np.stack((arc_length, lateral_dist), axis=1)
        # Set points beyond start or end of the line to nan
        cc[np.sum((0 <= s) & (s <= 1), axis=1) == 0] = np.nan
        curvilinear_coords.append(cc)
    return np.array(curvilinear_coords)


def merge_dicts_recursively(*dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                if isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = merge_dicts_recursively(result[k], v)
                else:
                    result[k] = v  # in case of two different values take the one later added
            else:
                result[k] = v
    return result
