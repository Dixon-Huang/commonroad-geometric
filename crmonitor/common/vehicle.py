import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Union, Dict, List, Tuple, Optional, Set

import numba
import numpy as np
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.trajectory import State
from shapely import affinity

from crmonitor.common.road_network import Lane, RoadNetwork

rot_mat_factors = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.]])

logger = logging.getLogger(__name__)


@numba.njit
def calc_s(s, w, l, theta):
    s = rot_mat_factors[0] * l / 2. * np.cos(theta) - rot_mat_factors[
        1] * w / 2 * np.sin(theta) + s
    return s


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """

    __slots__ = ["s", "v", "a", "j"]

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """

    __slots__ = ["d", "theta", "kappa", "kappa_dot"]

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


class Input:
    """
    Lateral and longitudinal vehicle input
    """

    __slots__ = ["a", "kappa_dot_dot"]

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


@dataclass
class CurvilinearStateManager:
    """
    Manage, cache curvilinear states
    """
    road_network: RoadNetwork
    cache: Dict[Tuple[State, Lane], Tuple[StateLongitudinal,
                                          StateLateral]] = field(default_factory=dict)

    @staticmethod
    def _compute_curvilinear_state(state: State, lane: Lane) -> Optional[Tuple[StateLongitudinal, StateLateral]]:
        try:
            s, d = lane.clcs.convert_to_curvilinear_coords(*state.position)
        except ValueError:
            logger.debug(
                "Vehicle out of projection domain: State will not be considered")
            return None
        theta_cl = lane.orientation(s)
        if hasattr(state, "acceleration") and hasattr(state, "jerk"):
            x_lon = StateLongitudinal(s=s, v=state.velocity,
                                      a=state.acceleration, j=state.jerk)
        elif hasattr(state, "acceleration"):
            x_lon = StateLongitudinal(s=s, v=state.velocity, a=state.acceleration)
        else:
            x_lon = StateLongitudinal(s=s, v=state.velocity)
        x_lat = StateLateral(d=d, theta=(state.orientation - theta_cl))
        return x_lon, x_lat

    def get_curvilinear_state(self, state: State, lane: Lane) -> Tuple[StateLongitudinal, StateLateral]:
        """

        :param state:
        :param lane: Reference lane
        :return:
        """
        key = (state.time_step, lane.lane_id)
        ccosy_state = self.cache.get(key)
        if ccosy_state is None:
            ccosy_state = self._compute_curvilinear_state(state, lane)
            self.cache[key] = ccosy_state
        return ccosy_state


@dataclass
class PredicateCache:
    # Levels: time step, predicate name, agent_ids
    cache: Dict[int, Dict[str, Dict[int, float]]] = field(
        default_factory=partial(defaultdict, partial(defaultdict, dict)))

    def get_robustness(self, time_step: int, predicate_name: str, other_ids: Union[Tuple[int], int]) -> Optional[float]:
        return self.cache[time_step][predicate_name].get(other_ids)

    def set_robustness(self, time_step: int, predicate_name: str, other_ids: Union[Tuple[int], int], robustness: float):
        self.cache[time_step][predicate_name][other_ids] = robustness

    def __contains__(self, item):
        assert isinstance(item, tuple) and len(item) == 3
        time_step, predicate_name, other_ids = item
        rob = self.cache[time_step][predicate_name].get(other_ids)
        return rob is not None

    def __getitem__(self, item):
        assert isinstance(item, tuple) and len(item) == 3
        time_step = item[0]
        predicate_name = item[1]
        ids = item[2]
        if isinstance(predicate_name, slice):
            # Only accept slice over all predicates
            assert predicate_name.start is None and predicate_name.stop is None and predicate_name.step is None
            return {n: pred_vals[ids] for n, pred_vals in self.cache[time_step].items() if len(pred_vals) > 0 and len(list(pred_vals.keys())[0]) == 1}
        else:
            return self.get_robustness(*item)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 3
        self.set_robustness(*key, value)


class Vehicle:
    def __init__(self, id, obstacle_type, vehicle_param, shape, states_cr, signal_series, ccosy_cache, lanelet_assignment, predicate_cache=None):
        self.id = id
        self.obstacle_type = obstacle_type
        self.vehicle_param = vehicle_param
        self.shape = shape
        self.states_cr: Dict = states_cr
        self.signal_series: Dict = signal_series
        self.ccosy_cache = ccosy_cache
        self.lanelet_assignment: Dict = lanelet_assignment
        self.predicate_cache = predicate_cache or PredicateCache()

    def rear_s(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates rear s-coordinate of vehicle

        :param time_step: time step to consider
        :returns rear s-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        curvi_state = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane)
        if curvi_state is None:
            return None
        state_lon, state_lat = curvi_state
        s = state_lon.s
        w = self.shape.width
        l = self.shape.length
        theta = state_lat.theta
        rear_s = np.min(calc_s(s, w, l, theta))
        return rear_s

    def front_s(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates front s-coordinate of vehicle

        :param time_step: time step to consider
        :returns front s-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        curvi_state = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane)
        if curvi_state is None:
            return None
        state_lon, state_lat = curvi_state
        s = state_lon.s
        w = self.shape.width
        l = self.shape.length
        theta = state_lat.theta
        front_s = np.max(calc_s(s, w, l, theta))
        return front_s

    def left_d(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates left d-coordinate of vehicle

        :param time_step: time step to consider
        :returns left d-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane)
        d = state_lat.d
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        return max(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def right_d(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates right d-coordinate of vehicle

        :param time_step: time step to consider
        :returns right d-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane)
        d = state_lat.d
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        return min(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def get_lat_state(self, time_step: int, lane: Lane = None):
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane)
        return state_lat

    def get_lon_state(self, time_step: int, lane: Lane = None):
        lane = lane or self.get_lane(time_step)
        states = self.ccosy_cache.get_curvilinear_state(self.states_cr[time_step], lane)
        return states[0] if states is not None else None

    def occupancy_at_time_step(self, time_step):
        state = self.states_cr[time_step]
        orientation = state.orientation
        shape = self.shape.rotate_translate_local(state.position,
                                                  orientation)
        return shape

    def shapely_occupancy_at_time_step(self, time_step):
        state = self.states_cr[time_step]
        orientation = state.orientation
        shape = self.shape.shapely_object
        cos = np.cos(orientation)
        sin = np.sin(orientation)
        mat = [cos, -sin, sin, cos, state.position[0], state.position[1]]
        new_shape = affinity.affine_transform(shape, mat)
        return new_shape

    def is_valid(self, time_step):
        state = self.states_cr.get(time_step)
        return state is not None

    def lanes_at_state(self, time_step) -> Set[Lane]:
        lanelets = self.lanelet_assignment[time_step]
        return self.ccosy_cache.road_network.find_lanes_by_lanelets(lanelets)

    def get_lane(self, time_step):
        # Todo: How to decide lane assignment generally?
        lanes = self.lanes_at_state(time_step)
        return lanes.pop() if len(lanes) > 0 else None

    @property
    def end_time(self):
        return max(map(lambda state: state.time_step, self.states_cr.values()))

    @property
    def start_time(self):
        return min(map(lambda state: state.time_step, self.states_cr.values()))

    @property
    def state_list_cr(self) -> List[State]:
        return list(self.states_cr.values())

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


class ControlledVehicle(Vehicle):

    def __init__(self, obstacle_id, vehicle_param, shape, road_network: RoadNetwork, inital_state: State,
                 obstacle_type=ObstacleType.CAR, initial_signal=None):
        states_cr = {inital_state.time_step: inital_state}
        signal_series = {inital_state.time_step: initial_signal}
        ccosy_cache = CurvilinearStateManager(road_network)
        self.lanelet_network = road_network.lanelet_network
        initial_lanelets = road_network.lanelet_network.find_lanelet_by_shape(
            shape.rotate_translate_local(inital_state.position, inital_state.orientation))
        lanelet_assignment = {inital_state.time_step: initial_lanelets}
        super().__init__(obstacle_id, obstacle_type, vehicle_param, shape, states_cr, signal_series, ccosy_cache,
                         lanelet_assignment)

    def add_state(self, state: State, signal_state=None):
        self.states_cr[state.time_step] = state
        loc_shape = self.shape.rotate_translate_local(state.position, state.orientation)
        self.lanelet_assignment[state.time_step] = self.lanelet_network.find_lanelet_by_shape(
            loc_shape)
        self.signal_series[state.time_step] = signal_state

    def remove_state(self, time_step):
        '''
        remove states before a given time step
        '''
        states_time_steps = [time for time in self.states_cr.keys()]
        for time in states_time_steps:
            if time < time_step:
                del self.states_cr[time]
                del self.lanelet_assignment[time]
                del self.signal_series[time]


class DynamicObstacleVehicle(Vehicle):
    """
    Representation of a vehicle with state and input profiles and other information for complete simulation horizon
    """

    def __init__(self, obstacle: DynamicObstacle, ccosy_cache: CurvilinearStateManager, vehicle_param, predicate_cache=None):
        lanelet_assignment = obstacle.prediction.shape_lanelet_assignment.copy()
        id = obstacle.obstacle_id
        obstacle_type = obstacle.obstacle_type
        vehicle_param = vehicle_param
        states_cr = {state.time_step: state for state in [
            obstacle.initial_state] + obstacle.prediction.trajectory.state_list}
        shape = obstacle.obstacle_shape
        if obstacle.signal_series is not None:
            signal_series = {state.time_step: state for state in obstacle.signal_series}
        else:
            signal_series = None
        ccosy_cache = ccosy_cache
        lanelet_assignment[obstacle.initial_state.time_step] = obstacle.initial_shape_lanelet_ids
        super().__init__(id, obstacle_type, vehicle_param, shape, states_cr,
                         signal_series, ccosy_cache, lanelet_assignment, predicate_cache)

    # @property
    # def states_lon(self) -> Dict[int, StateLongitudinal]:
    #     return self._states_lon
    #
    # @property
    # def states_lat(self) -> Dict[int, StateLateral]:
    #     return self._states_lat
