import logging
import numpy as np
from typing import List, Tuple, Set, Iterable, Dict, Callable, Union

from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.lanelet import (
    LaneletType,
    LineMarking,
    Lanelet,
    LaneletNetwork,
)

from crmonitor.common.helper import cartesian_to_curvilinear
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.common.road_network import RoadNetwork

logger = logging.getLogger(__name__)


def distance_to_bounds(
    vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step
):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle_i.shape.vertices[:-1], state.position, state.orientation
    )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i) for i in lanelet_ids
    ]
    left_bounds = tuple(
        [
            l.left_vertices
            for l in lanelets
            if l.adj_left is not None and l.adj_left not in lanelet_ids
        ]
    )
    right_bounds = tuple(
        [
            l.right_vertices
            for l in lanelets
            if l.adj_right is not None and l.adj_right not in lanelet_ids
        ]
    )
    if len(left_bounds) > 0:
        d_left = np.array(cartesian_to_curvilinear(left_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_left = d_left[~np.isnan(d_left)]
    else:
        d_left = np.array([])
    if len(right_bounds) > 0:
        d_right = np.array(cartesian_to_curvilinear(right_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_right = d_right[~np.isnan(d_right)]
    else:
        d_right = np.array([])

    return d_left, d_right


def distance_to_lanes(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world, time_step):
    d_left, d_right = distance_to_bounds(vehicle_i, lanelet_ids, world, time_step)
    d_left = -np.min(d_left) if d_left.size > 0 else np.inf
    d_right = np.max(d_right) if d_right.size > 0 else np.inf
    return np.fmin(d_left, d_right)


def lanelets_left_of_lanelet(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelet IDs left of a given lanelet based on adjacency relations

    :param lanelet: given lanelet
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    left_lanelets = set()
    tmp_lanelet = lanelet
    while tmp_lanelet.adj_left is not None:
        tmp_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_left)
        left_lanelets.add(tmp_lanelet)

    return left_lanelets


def lanelets_right_of_lanelet(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelet IDs right of a given lanelet based on adjacency relations

    :param lanelet: given lanelet
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    right_lanelets = set()
    tmp_lanelet = lanelet
    while tmp_lanelet.adj_right is not None:
        tmp_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_right)
        right_lanelets.add(tmp_lanelet)

    return right_lanelets


def lanelets_left_of_vehicle(
    time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelets left of a vehicle

    :param vehicle: vehicle of interest
    :param time_step: time step of interest
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    left_lanelets = set()
    occupied_lanelets = vehicle.lanelet_assignment[time_step]
    for occ_l in occupied_lanelets:
        new_lanelets = lanelets_left_of_lanelet(
            lanelet_network.find_lanelet_by_id(occ_l), lanelet_network
        )
        for lanelet in new_lanelets:
            left_lanelets.add(lanelet)

    return left_lanelets


def lanelets_right_of_vehicle(
    time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelets right of a vehicle

    :param vehicle: vehicle of interest
    :param time_step: time step of interest
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    right_lanelets = set()
    occupied_lanelets = vehicle.lanelet_assignment[time_step]
    for occ_l in occupied_lanelets:
        new_lanelets = lanelets_right_of_lanelet(
            lanelet_network.find_lanelet_by_id(occ_l), lanelet_network
        )
        for lanelet in new_lanelets:
            right_lanelets.add(lanelet)

    return right_lanelets


def vehicles_adjacent(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
     Searches for vehicles adjacent to a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of adjacent vehicles of a vehicle
    """
    vehicles_adj = []
    lane_share = vehicle.get_lane(time_step)
    for veh in other_vehicles:
        if veh.get_lon_state(time_step, lane_share) is None:
            continue
        if (
            veh.rear_s(time_step, lane_share)
            < vehicle.front_s(time_step, lane_share)
            < veh.front_s(time_step, lane_share)
        ):
            vehicles_adj.append(veh)
            continue
        if (
            veh.rear_s(time_step, lane_share)
            < vehicle.rear_s(time_step, lane_share)
            < veh.front_s(time_step, lane_share)
        ):
            vehicles_adj.append(veh)
            continue
        if vehicle.rear_s(time_step, lane_share) <= veh.rear_s(
            time_step, lane_share
        ) and veh.front_s(time_step, lane_share) <= vehicle.front_s(
            time_step, lane_share
        ):
            vehicles_adj.append(veh)
            continue
    return vehicles_adj


def vehicles_left(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
    Searches for vehicles left of a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of vehicles left of a vehicle
    """
    vehicles_adj = vehicles_adjacent(time_step, vehicle, other_vehicles)
    lane_share = vehicle.get_lane(time_step)
    vehicles_left = [
        veh
        for veh in vehicles_adj
        if veh.right_d(time_step, lane_share) > vehicle.left_d(time_step, lane_share)
    ]
    return vehicles_left


def vehicle_directly_left(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> Union[Vehicle, None]:
    vehicle_left = vehicles_left(time_step, vehicle, other_vehicles)
    if len(vehicle_left) == 0:
        return None
    elif len(vehicle_left) == 1:
        return vehicle_left[0]
    else:
        vehicle_directly_left = vehicle_left[0]
        for veh in vehicle_left:
            lane_share = veh.get_lane(time_step)
            if (
                veh.get_lat_state(time_step, lane_share).d
                < vehicle_directly_left.get_lat_state(time_step, lane_share).d
            ):
                vehicle_directly_left = veh
        return vehicle_directly_left


def vehicles_right(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
    Searches for vehicles right of a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of vehicles left of a vehicle
    """
    vehicles_adj = vehicles_adjacent(time_step, vehicle, other_vehicles)
    lane_share = vehicle.get_lane(time_step)
    vehicles_right = [
        veh
        for veh in vehicles_adj
        if veh.left_d(time_step, lane_share) < vehicle.right_d(time_step, lane_share)
    ]
    return vehicles_right


def vehicle_directly_right(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> Union[Vehicle, None]:
    vehicle_right = vehicles_right(time_step, vehicle, other_vehicles)
    if len(vehicle_right) == 0:
        return None
    elif len(vehicle_right) == 1:
        return vehicle_right[0]
    else:
        vehicle_directly_right = vehicle_right[0]
        for veh in vehicle_right:
            lane_share = veh.get_lane(time_step)
            if (
                veh.get_lat_state(time_step, lane_share).d
                > vehicle_directly_right.get_lat_state(time_step, lane_share).d
            ):
                vehicle_directly_right = veh
        return vehicle_directly_right


def _adjacent_lanelets(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Returns all lanelet which are adjacent to a lanelet and the lanelet itself

    :param lanelet: CommonRoad lanelet
    :returns set of adjacent lanelets
    """
    lanelets = {lanelet}
    la = lanelet
    while la is not None and la.adj_left is not None:
        la = lanelet_network.find_lanelet_by_id(la.adj_left)
        if la is not None:
            lanelets.add(la)
    la = lanelet
    while la is not None and la.adj_right is not None:
        la = lanelet_network.find_lanelet_by_id(la.adj_right)
        if la is not None:
            lanelets.add(la)
    return lanelets


def cal_road_width(
    lanelet: Lanelet, road_network: RoadNetwork, position: float
) -> float:
    """
    Calculates width of road given a lanelet and a longitudinal position
    """
    adj_lanelets = _adjacent_lanelets(lanelet, road_network.lanelet_network)
    road_width = 0.0
    for lanelet in list(adj_lanelets):
        road_width += road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(
            position
        )
    return road_width
