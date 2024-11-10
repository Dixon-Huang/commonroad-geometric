from enum import Enum
import logging
import math
import operator
from typing import List, Tuple, Set, Dict, Callable
from shapely.geometry.polygon import Polygon
import numpy as np

from commonroad.scenario.lanelet import LaneletType, LineMarking

from ruamel.yaml.comments import CommentedMap

from crmonitor.common.helper import union_set
from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

from crmonitor.predicates.base import BasePredicateEvaluator, MAX_LONG_DIST
from crmonitor.predicates.utils import (
    distance_to_bounds,
    distance_to_lanes,
    lanelets_left_of_vehicle,
    lanelets_right_of_vehicle,
    vehicle_directly_left,
    vehicle_directly_right,
)

logger = logging.getLogger(__name__)


class PositionPredicates(str, Enum):
    InSameLane = "in_same_lane"
    InFrontOf = "in_front_of"
    SingleLane = "single_lane"
    KeepsSafeDistancePrec = "keeps_safe_distance_prec"
    Precedes = "precedes"
    # newly added besides the ones for R_G1-R_G3
    RightOfBroadLaneMarking = "right_of_broad_lane_marking"
    LeftOfBroadLaneMarking = "left_of_broad_lane_marking"
    OnAccessRamp = "on_access_ramp"
    OnShoulder = "on_shoulder"
    OnMainCarriageway = "on_main_carriage_way"
    OnExitRamp = "on_exit_ramp"  # not used
    InRightmostLane = "in_rightmost_lane"
    InLeftmostLane = "in_leftmost_lane"
    MainCarriagewayRightLane = "main_carriageway_right_lane"
    LeftOf = "left_of"
    DrivesLeftmost = "drives_leftmost"
    DrivesRightmost = "drives_rightmost"


class PredInSameLane(BasePredicateEvaluator):
    predicate_name = PositionPredicates.InSameLane
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        intersecting_lanes = self.get_same_lanes(world, time_step, vehicle_ids)
        return len(intersecting_lanes) > 0

    def get_same_lanes(self, world: World, time_step, vehicle_ids) -> Set[Lane]:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        lanes_k = world.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[time_step]
        )
        lanes_p = world.road_network.find_lanes_by_lanelets(
            vehicle_p.lanelet_assignment[time_step]
        )
        intersecting_lanes = lanes_p.intersection(lanes_k)
        return intersecting_lanes

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        If boolean is
        True: Minimum lateral displacement to not be in the same lane anymore
        False: Minimum distance to lanes of other
        """
        # Predicate is symmetric
        vehicle_ids_tuple = tuple(reversed(vehicle_ids))
        value = world.vehicle_by_id(vehicle_ids_tuple[0]).predicate_cache[
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        ]
        if value is not None:
            return value

        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelet_ids_k = union_set(
            [l.contained_lanelets for l in vehicle_k.lanes_at_state(time_step)]
        )
        lanelet_ids_p = union_set(
            [l.contained_lanelets for l in vehicle_p.lanes_at_state(time_step)]
        )
        rob = np.fmin(
            distance_to_lanes(vehicle_k, lanelet_ids_p, world, time_step),
            distance_to_lanes(vehicle_p, lanelet_ids_k, world, time_step),
        )
        return self._scale_lat_dist(rob)


class PredInFrontOf(BasePredicateEvaluator):
    predicate_name = PositionPredicates.InFrontOf
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        rear = world.vehicle_by_id(vehicle_ids[0])
        front = world.vehicle_by_id(vehicle_ids[1])
        assert rear.front_s(time_step) is not None
        return self._scale_lon_dist(
            front.rear_s(time_step, rear.get_lane(time_step)) - rear.front_s(time_step)
        )


class PredSingleLane(BasePredicateEvaluator):
    predicate_name = PositionPredicates.SingleLane
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lanes = world.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[time_step]
        )
        return len(k_lanes) == 1

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        If false: 1 - the largest fractional overlap with occupied lanes
        If true: Distance to lane polygon boundary
        """
        # single_lane_boolean = self.evaluate_boolean(world, vehicle_ids)
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lanes = sorted(vehicle_k.lanes_at_state(time_step))
        assert len(k_lanes) > 0, (
            f"Vehicle must be assigned to at least one lane! "
            f"{str(world.scenario.scenario_id)}, "
            f"id={vehicle_ids[0]}, t={time_step}"
        )

        ref_point = np.array(vehicle_k.states_cr[time_step].position)
        ref_lanes = [l for l in k_lanes if l.lanelet.polygon.contains_point(ref_point)]
        ref_lane = ref_lanes[0] if len(ref_lanes) > 0 else k_lanes[0]

        d_left, d_right = distance_to_bounds(
            vehicle_k, ref_lane.contained_lanelets, world, time_step
        )
        d_left = -np.max(d_left) if d_left.size > 0 else np.inf
        d_right = np.min(d_right) if d_right.size > 0 else np.inf
        rob = np.fmin(d_left, d_right)
        return self._scale_lat_dist(rob)


class PredSafeDistPrec(BasePredicateEvaluator):
    predicate_name = PositionPredicates.KeepsSafeDistancePrec
    arity = 2

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def calculate_safe_distance(
        cls, v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow
    ):
        d_safe = (
            (v_lead**2) / (-2 * np.abs(a_min_lead))
            - (v_follow**2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )

        return d_safe

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world.vehicle_by_id(vehicle_ids[1])
        time_step = time_step

        if vehicle_lead.get_lane(time_step) is None:
            return self._scale_lon_dist(math.inf)
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = self.calculate_safe_distance(
            vehicle_follow.states_cr[time_step].velocity,
            vehicle_lead.states_cr[time_step].velocity,
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(time_step)
        rob = self._scale_lon_dist(delta_s - safe_distance)
        return rob

    @staticmethod
    def _plot_red_arrow(ax, x, y, size=1.0):
        ax.plot(x, y, linewidth=2, color="r", zorder=25)
        ax.arrow(
            x[-2],
            y[-2],
            x[-1] - x[-2],
            y[-1] - y[-2],
            lw=0,
            length_includes_head=True,
            head_width=size,
            head_length=size,
            zorder=25,
            color="r",
        )

    def visualize_unsafe_region(
        self, ax, time_step: int, unsafe_s: float, vehicle_lead: Vehicle
    ):
        """
        Plots the unsafe region starting from the rear of the front vehicle
        """
        # the ids of lanes are increasing together with the d-coordinate
        vehicle_lanes = list(vehicle_lead.lanes_at_state(time_step))
        # the upper the lane in the road network is, the smaller the index in the list as
        if (
            vehicle_lanes[0].lanelet.center_vertices[0][1]
            < vehicle_lanes[-1].lanelet.center_vertices[0][1]
        ):
            vehicle_lanes = vehicle_lanes[::-1]
        reference_lane = vehicle_lead.get_lane(time_step)
        # get the Cartesian coordinate of the safe distance
        safe_pos_cart = reference_lane.clcs.convert_to_cartesian_coords(unsafe_s, 0)
        lead_rear_cart = reference_lane.clcs.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        # left vertices
        front_rear_left_cart = vehicle_lanes[0].clcs_left.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        safe_pos_left_cart = vehicle_lanes[0].clcs_left.convert_to_cartesian_coords(
            unsafe_s, 0
        )
        reference_left = np.vstack(vehicle_lanes[0].clcs_left.reference_path())
        vertices_left = reference_left[
            (reference_left[:, 0] > safe_pos_left_cart[0])
            & (reference_left[:, 0] < front_rear_left_cart[0]),
            :,
        ]
        vertices_left = np.concatenate(
            ([safe_pos_left_cart], vertices_left, [front_rear_left_cart])
        )
        # right vertices
        lead_rear_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        safe_pos_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(
            unsafe_s, 0
        )
        reference_right = np.vstack(vehicle_lanes[-1].clcs_right.reference_path())
        vertices_right = reference_right[
            (reference_right[:, 0] > safe_pos_left_cart[0])
            & (reference_right[:, 0] < front_rear_left_cart[0]),
            :,
        ]
        vertices_right = np.concatenate(
            ([safe_pos_right_cart], vertices_right, [lead_rear_right_cart])
        )
        # concatenate vertices
        vertices_total = list(
            np.concatenate(
                (
                    [safe_pos_cart],
                    vertices_left,
                    [lead_rear_cart],
                    vertices_right,
                    [safe_pos_cart],
                )
            )
        )
        # compute centroid
        cent = (
            sum([v[0] for v in vertices_total]) / len(vertices_total),
            sum([v[1] for v in vertices_total]) / len(vertices_total),
        )
        # sort by polar angle
        vertices_total.sort(key=lambda v: math.atan2(v[1] - cent[1], v[0] - cent[0]))

        unsafe_region = Polygon(vertices_total)
        ax.fill(
            *unsafe_region.exterior.xy,
            zorder=30,
            alpha=0.2,
            facecolor="red",
            edgecolor=None,
        )

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        latest_value = self.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )
        latest_value_unscaled = (
            latest_value * MAX_LONG_DIST
        )  # un-scale to actual range and make positive
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])

        lane_clcs = vehicle_follow.get_lane(
            time_step
        ).clcs  # center curvilinear coordinate system
        sampling_step_size = 1.0

        s_start = vehicle_follow.front_s(time_step)
        num_points = max(2, abs(int(latest_value_unscaled / sampling_step_size)))
        points_s = np.linspace(0, latest_value_unscaled, num_points) + s_start
        points_s = points_s[:, None]
        points_l = np.zeros((points_s.shape[0], 1))
        points_curvi = np.concatenate((points_s, points_l), axis=1)
        points_cartesian = np.stack(
            [lane_clcs.convert_to_cartesian_coords(*p) for p in points_curvi], axis=0
        )

        def fun(renderer):
            self._plot_red_arrow(
                renderer.ax, points_cartesian[:, 0], points_cartesian[:, 1]
            )
            unsafe_s = latest_value_unscaled + s_start
            self.visualize_unsafe_region(
                renderer.ax, time_step, unsafe_s, world.vehicle_by_id(vehicle_ids[1])
            )

        return (fun,)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((0, 1))
        ax.plot(0, 0.5, color="r")
        PredSafeDistPrec._plot_red_arrow(ax, [0, 1], [0.5, 0.5], size=0.1)
        PredSafeDistPrec._plot_red_arrow(ax, [0, -1], [0.5, 0.5], size=0.1)


class PredPreceding(BasePredicateEvaluator):
    predicate_name = PositionPredicates.Precedes
    arity = 2

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.same_lane = PredInSameLane(config)

    @staticmethod
    def get_predecessors(
        world: World, time_step, vehicle_rear: Vehicle
    ) -> List[Tuple[float, Vehicle, Lane, bool]]:
        """
        Returns a list of preceding vehicles in ascending order of distance
        :param time_step:
        :param world: Current world state
        :param vehicle_rear: Reference vehicle
        :return: Sorted list of tuples of distance and vehicle object
        """
        veh = []
        rear_lanes = vehicle_rear.lanes_at_state(time_step)
        for vehicle_front in world.vehicles:
            if not vehicle_front.is_valid(time_step) or vehicle_front is vehicle_rear:
                continue
            front_lanes = vehicle_front.lanes_at_state(time_step)
            intersecting_lanes = rear_lanes.intersection(front_lanes)

            # Change
            if not intersecting_lanes:  # 如果没有共同车道,直接跳过
                continue

            same_lane = len(intersecting_lanes) > 0

            lane = (
                list(intersecting_lanes)[0]
                if len(intersecting_lanes) > 0
                else vehicle_rear.get_lane(time_step)
            )

            dist = vehicle_front.rear_s(time_step, lane) - vehicle_rear.front_s(
                time_step, lane
            )
            veh.append((dist, vehicle_front, lane, same_lane))
        return sorted(veh, key=lambda d: d[0])

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        rear_vehicle_id = vehicle_ids[0]
        front_vehicle_id = vehicle_ids[1]
        rear_vehicle = world.vehicle_by_id(rear_vehicle_id)
        pred_veh = self.get_predecessors(world, time_step, rear_vehicle)
        return len(pred_veh) > 0 and pred_veh[0][1].id == front_vehicle_id

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        rear_veh = world.vehicle_by_id(vehicle_ids[0])
        front_veh = world.vehicle_by_id(vehicle_ids[1])
        veh_lon_dist = self.get_predecessors(world, time_step, rear_veh)
        veh_front_dist = [_ for _ in veh_lon_dist if _[0] >= 0 and _[3]]
        bool_val = len(veh_front_dist) > 0 and veh_front_dist[0][1].id == vehicle_ids[1]
        same_lane = self.same_lane.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )
        if bool_val:
            assert same_lane >= -self.eps
            same_lane = max(same_lane, 0.0)

        for dist, veh, _, __ in veh_lon_dist:
            if veh == front_veh:
                dist_front = dist
                break
        else:
            # Should never happen
            assert False

        pred_wo_other = [v for v in veh_front_dist if v[1] is not front_veh]
        if len(pred_wo_other) > 0:
            _, pred_wo_other, lane, __ = pred_wo_other[0]
            dist_pred = pred_wo_other.rear_s(time_step, lane) - front_veh.rear_s(
                time_step
            )
        else:
            dist_pred = math.inf

        rob = min(
            same_lane,
            self._scale_lon_dist(dist_front),
            self._scale_lon_dist(dist_pred),
        )
        return rob


class PredRightOfBroadLaneMarking(BasePredicateEvaluator):
    predicate_name = PositionPredicates.RightOfBroadLaneMarking
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                return False

        lanelets_left_of_veh = lanelets_left_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_left_of_veh:
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                _, d_right = distance_to_bounds(vehicle, [l_id], world, time_step)
                d_right = np.min(d_right) if d_right.size > 0 else np.inf
                # the abs function is to prevent the distance to be positive but the robustness is violated
                return self._scale_lat_dist(-abs(d_right))
        lanelets_left_of_veh = lanelets_left_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_left_of_veh:
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                d_left, _ = distance_to_bounds(
                    vehicle, [lanelet.lanelet_id], world, time_step
                )
                d_left = -np.max(d_left) if d_left.size > 0 else np.inf
                return self._scale_lat_dist(d_left)
        return self._scale_lat_dist(-np.inf)


class PredLeftOfBroadLaneMarking(BasePredicateEvaluator):
    predicate_name = PositionPredicates.LeftOfBroadLaneMarking
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                return False

        lanelets_right_of_veh = lanelets_right_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_right_of_veh:
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                d_left, _ = distance_to_bounds(vehicle, [l_id], world, time_step)
                d_left = -np.max(d_left) if d_left.size > 0 else np.inf
                # the abs function is to prevent the distance to be positive but the robustness is violated
                return self._scale_lat_dist(-abs(d_left))
        lanelets_right_of_veh = lanelets_right_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_right_of_veh:
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                _, d_right = distance_to_bounds(
                    vehicle, [lanelet.lanelet_id], world, time_step
                )
                d_right = np.min(d_right) if d_right.size > 0 else np.inf
                return self._scale_lat_dist(d_right)
        return self._scale_lat_dist(-np.inf)


class PredOnAccessRamp(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is on an access ramp.
    """

    predicate_name = PositionPredicates.OnAccessRamp
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        access_ramp_ids = [
            l_id
            for l_id in lanelet_ids_occ
            if LaneletType.ACCESS_RAMP
            in world.road_network.lanelet_network.find_lanelet_by_id(l_id).lanelet_type
        ]
        if len(access_ramp_ids) > 0:
            return self._scale_lat_dist(
                distance_to_lanes(vehicle, access_ramp_ids, world, time_step)
            )
        else:
            return self._scale_lat_dist(-np.inf)


class PredOnShoulder(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is on a shoulder lane.
    """

    predicate_name = PositionPredicates.OnShoulder
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.SHOULDER in lanelet.lanelet_type:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        shoulder_ids = [
            l_id
            for l_id in lanelet_ids_occ
            if LaneletType.SHOULDER
            in world.road_network.lanelet_network.find_lanelet_by_id(l_id).lanelet_type
        ]
        if len(shoulder_ids) > 0:
            return self._scale_lat_dist(
                distance_to_lanes(vehicle, shoulder_ids, world, time_step)
            )
        else:
            return self._scale_lat_dist(-np.inf)


class PredOnMainCarriageway(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is on a main carriage way.
    """

    predicate_name = PositionPredicates.OnMainCarriageway
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        main_carriage_way_ids = [
            l_id
            for l_id in lanelet_ids_occ
            if LaneletType.MAIN_CARRIAGE_WAY
            in world.road_network.lanelet_network.find_lanelet_by_id(l_id).lanelet_type
        ]
        if len(main_carriage_way_ids) > 0:
            return self._scale_lat_dist(
                distance_to_lanes(vehicle, main_carriage_way_ids, world, time_step)
            )
        else:
            return self._scale_lat_dist(-np.inf)


class PredInRightmostLane(BasePredicateEvaluator):
    """
    check if any assigned lanelet of ego vehicle is in rightmost lane
    """

    predicate_name = PositionPredicates.InRightmostLane
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.adj_right_same_direction is None:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        rightmost_lanelet_ids = [
            l.lanelet_id
            for l in world.road_network.lanelet_network.lanelets
            if l.adj_right_same_direction is None
        ]
        dis_to_lane = distance_to_lanes(
            vehicle, rightmost_lanelet_ids, world, time_step
        )
        return self._scale_lat_dist(dis_to_lane)


class PredInLeftmostLane(BasePredicateEvaluator):
    """
    check if any assigned lanelet of ego vehicle is in leftmost lane
    """

    predicate_name = PositionPredicates.InLeftmostLane
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.adj_left_same_direction is None:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        leftmost_lanelet_ids = [
            l.lanelet_id
            for l in world.road_network.lanelet_network.lanelets
            if l.adj_left_same_direction is None
        ]
        dis_to_lane = distance_to_lanes(vehicle, leftmost_lanelet_ids, world, time_step)
        return self._scale_lat_dist(dis_to_lane)


class PredMainCarriageWayRightLane(BasePredicateEvaluator):
    """
    Evaluates if a vehicle occupies the rightmost main carriageway lane.
    """

    predicate_name = PositionPredicates.MainCarriagewayRightLane
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type and (
                not lanelet.adj_right_same_direction
                or LaneletType.MAIN_CARRIAGE_WAY
                not in world.road_network.lanelet_network.find_lanelet_by_id(
                    lanelet.adj_right
                ).lanelet_type
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        main_carriageway_right_lanelet_ids = [
            l.lanelet_id
            for l in world.road_network.lanelet_network.lanelets
            if LaneletType.MAIN_CARRIAGE_WAY in l.lanelet_type
            and (
                not l.adj_right_same_direction
                or LaneletType.MAIN_CARRIAGE_WAY
                not in world.road_network.lanelet_network.find_lanelet_by_id(
                    l.adj_right
                ).lanelet_type
            )
        ]
        dis_to_lane = distance_to_lanes(
            vehicle, main_carriageway_right_lanelet_ids, world, time_step
        )
        return self._scale_lat_dist(dis_to_lane)


class PredLeftOf(BasePredicateEvaluator):
    predicate_name = PositionPredicates.LeftOf
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        Evaluates if the kth vehicle is left of the pth vehicle
        """
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        # share the same lane as the reference, otherwise the comparison does not make sense
        lane_share = list(
            world.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[time_step]
            )
        )[0]
        if not vehicle_p.left_d(time_step, lane_share) < vehicle_k.right_d(
            time_step, lane_share
        ):
            return False
        else:
            if (
                vehicle_p.rear_s(time_step, lane_share)
                <= vehicle_k.front_s(time_step, lane_share)
                <= vehicle_p.front_s(time_step, lane_share)
            ):
                return True
            if (
                vehicle_p.rear_s(time_step, lane_share)
                <= vehicle_k.rear_s(time_step, lane_share)
                <= vehicle_p.front_s(time_step, lane_share)
            ):
                return True
            if vehicle_k.rear_s(time_step, lane_share) < vehicle_p.rear_s(
                time_step, lane_share
            ) and vehicle_p.front_s(time_step, lane_share) < vehicle_k.front_s(
                time_step, lane_share
            ):
                return True
            else:
                return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        # share the same lane as the reference, otherwise the comparison does not make sense
        lane_share = list(
            world.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[time_step]
            )
        )[0]
        left_p = vehicle_p.left_d(time_step, lane_share)
        right_k = vehicle_k.right_d(time_step, lane_share)
        rear_p = vehicle_p.rear_s(time_step, lane_share)
        rear_k = vehicle_k.rear_s(time_step, lane_share)
        front_p = vehicle_p.front_s(time_step, lane_share)
        front_k = vehicle_k.front_s(time_step, lane_share)

        #  pred = (left_p < right_k) and (
        #         (rear_p <= rear_k and rear_k <= front_p) or (front_p < front_k and rear_k < rear_p) or (
        #         rear_p <= front_k and front_k <= front_p))

        left_p_less_than_right_k = self._scale_lat_dist(right_k - left_p)
        rear_p_less_than_rear_k = self._scale_lon_dist(rear_k - rear_p)
        rear_k_less_than_front_p = self._scale_lon_dist(front_p - rear_k)
        front_p_less_than_front_k = self._scale_lon_dist(front_k - front_p)
        rear_k_less_than_rear_p = self._scale_lon_dist(rear_p - rear_k)
        rear_p_less_than_front_k = self._scale_lon_dist(front_k - rear_p)
        front_k_less_than_front_p = self._scale_lon_dist(front_p - front_k)

        rob = min(
            left_p_less_than_right_k,
            max(
                min(rear_p_less_than_rear_k, rear_k_less_than_front_p),
                min(front_p_less_than_front_k, rear_k_less_than_rear_p),
                min(rear_p_less_than_front_k, front_k_less_than_front_p),
            ),
        )
        return rob


class PredDrivesLeftmost(BasePredicateEvaluator):
    """
    Evaluates if a vehicle drives leftmost within its occupied lanes.
    """

    predicate_name = PositionPredicates.DrivesLeftmost
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        veh_dir_l = vehicle_directly_left(time_step, vehicle, other_vehicles)
        if veh_dir_l is not None:
            share_lane = vehicle.get_lane(time_step)
            if (
                veh_dir_l.right_d(time_step, share_lane)
                - vehicle.left_d(time_step, share_lane)
                < self.config["close_to_other_vehicle"]
            ):
                return True
            else:
                return False
        else:
            lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)
            for lane in lanes:
                left_position = vehicle.left_d(time_step, lane)
                s_ego = vehicle.get_lon_state(time_step, lane).s
                if (
                    0.5 * lane.width(s_ego) - left_position
                    > self.config["close_to_lane_border"]
                ):
                    return False
            return True

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        veh_dir_l = vehicle_directly_left(time_step, vehicle, other_vehicles)
        if veh_dir_l is not None:
            share_lane = vehicle.get_lane(time_step)
            return self._scale_lat_dist(
                self.config["close_to_other_vehicle"]
                - veh_dir_l.right_d(time_step, share_lane)
                + vehicle.left_d(time_step, share_lane)
            )
        else:
            lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)
            comparison_list = []  # the 'or' relations between different lanes
            for lane in lanes:
                left_position = vehicle.left_d(time_step, lane)
                s_ego = vehicle.get_lon_state(time_step, lane).s
                comparison_list.append(
                    self._scale_lat_dist(
                        self.config["close_to_lane_border"]
                        - 0.5 * lane.width(s_ego)
                        + left_position
                    )
                )
            return min(comparison_list)


class PredDrivesRightmost(BasePredicateEvaluator):
    """
    Evaluates if a vehicle drives rightmost within its occupied lanes.
    """

    predicate_name = PositionPredicates.DrivesRightmost
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        veh_dir_r = vehicle_directly_right(time_step, vehicle, other_vehicles)
        if veh_dir_r is not None:
            share_lane = vehicle.get_lane(time_step)
            if (
                -veh_dir_r.left_d(time_step, share_lane)
                + vehicle.right_d(time_step, share_lane)
                < self.config["close_to_other_vehicle"]
            ):
                return True
            else:
                return False
        else:
            lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)
            for lane in lanes:
                right_position = vehicle.right_d(time_step, lane)
                s_ego = vehicle.get_lon_state(time_step, lane).s
                if (
                    0.5 * lane.width(s_ego) + right_position
                    > self.config["close_to_lane_border"]
                ):
                    return False
            return True

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        veh_dir_r = vehicle_directly_right(time_step, vehicle, other_vehicles)
        if veh_dir_r is not None:
            share_lane = vehicle.get_lane(time_step)
            return self._scale_lat_dist(
                self.config["close_to_other_vehicle"]
                + veh_dir_r.left_d(time_step, share_lane)
                - vehicle.right_d(time_step, share_lane)
            )
        else:
            lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)
            comparison_list = []  # the 'or' relations between different lanes
            for lane in lanes:
                right_position = vehicle.right_d(time_step, lane)
                s_ego = vehicle.get_lon_state(time_step, lane).s
                comparison_list.append(
                    self._scale_lat_dist(
                        self.config["close_to_lane_border"]
                        - 0.5 * lane.width(s_ego)
                        - right_position
                    )
                )
            return min(comparison_list)
