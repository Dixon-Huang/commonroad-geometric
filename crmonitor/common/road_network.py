from typing import List, Set, Dict

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet, LaneletType
from commonroad_dc.geometry.util import chaikins_corner_cutting, \
    resample_polyline
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem


class Lane:
    """
    Lane representation build from several lanelets
    """

    def __init__(
        self,
        merged_lanelet: Lanelet,
        contained_lanelets: List[int],
        road_network_param: Dict,
    ):
        """
        :param merged_lanelet: lanelet element of lane
        :param contained_lanelets: lanelets lane consists of
        :param road_network_param: dictionary with parameters for the road network
        """
        self._lanelet = merged_lanelet
        self._contained_lanelets = set(contained_lanelets)
        self.lane_id = int("".join((str(i) for i in self._contained_lanelets)))
        self.clcs_left = Lane.create_curvilinear_coordinate_system_from_reference(
            merged_lanelet.left_vertices, road_network_param
        )
        self.clcs_right = Lane.create_curvilinear_coordinate_system_from_reference(
            merged_lanelet.right_vertices, road_network_param
        )
        self._clcs = Lane.create_curvilinear_coordinate_system_from_reference(
            merged_lanelet.center_vertices, road_network_param
        )
        self._orientation = self._compute_orientation_from_polyline(
            merged_lanelet.center_vertices
        )
        self._curvature = self._compute_curvature_from_polyline(
            merged_lanelet.center_vertices
        )
        self._path_length = self._compute_path_length_from_polyline(
            merged_lanelet.center_vertices
        )
        self._width = self._compute_width_from_lanalet_boundary(
            merged_lanelet.left_vertices, merged_lanelet.right_vertices
        )

    def __lt__(self, other):
        assert isinstance(other, Lane)
        return tuple(sorted(self.contained_lanelets)) < tuple(sorted(other.contained_lanelets))

    @property
    def lanelet(self) -> Lanelet:
        return self._lanelet

    @property
    def contained_lanelets(self) -> Set[int]:
        return self._contained_lanelets

    @property
    def clcs(self) -> CurvilinearCoordinateSystem:
        return self._clcs

    def orientation(self, position) -> float:
        """
        Calculates orientation of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns orientation of lane at a given position
        """
        return np.interp(position, self._path_length, self._orientation)

    def width(self, s_position: float) -> float:
        """
        Calculates width of lane given a longitudinal position along lane

        :param s_position: longitudinal position
        :returns width of lane at a given position
        """
        return np.interp(s_position, self._path_length, self._width)

    @staticmethod
    def _compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes orientation along a polyline

        :param polyline: polyline for which orientation should be calculated
        :return: orientation along polyline
        """
        assert (
            isinstance(polyline, np.ndarray)
            and len(polyline) > 1
            and polyline.ndim == 2
            and len(polyline[0, :]) == 2
        ), "<Math>: not a valid polyline. polyline = {}".format(polyline)
        if len(polyline) < 2:
            raise ValueError("Cannot create orientation from polyline of length < 2")

        orientation = [0]
        for i in range(1, len(polyline)):
            pt1 = polyline[i - 1]
            pt2 = polyline[i]
            tmp = pt2 - pt1
            orientation.append(np.arctan2(tmp[1], tmp[0]))

        return np.array(orientation)

    @staticmethod
    def _compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes curvature along a polyline

        :param polyline: polyline for which curvature should be calculated
        :return: curvature along  polyline
        """
        assert (
            isinstance(polyline, np.ndarray)
            and polyline.ndim == 2
            and len(polyline[:, 0]) > 2
        ), "Polyline malformed for curvature computation p={}".format(polyline)

        x_d = np.gradient(polyline[:, 0])
        x_dd = np.gradient(x_d)
        y_d = np.gradient(polyline[:, 1])
        y_dd = np.gradient(y_d)

        return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3.0 / 2.0))

    @staticmethod
    def _compute_path_length_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes the path length of a polyline

        :param polyline: polyline for which path length should be calculated
        :return: path length along polyline
        """
        assert (
            isinstance(polyline, np.ndarray)
            and polyline.ndim == 2
            and len(polyline[:, 0]) > 2
        ), "Polyline malformed for pathlenth computation p={}".format(polyline)

        distance = np.zeros((len(polyline),))
        for i in range(1, len(polyline)):
            distance[i] = distance[i - 1] + np.linalg.norm(
                polyline[i] - polyline[i - 1]
            )

        return np.array(distance)

    @staticmethod
    def _compute_width_from_lanalet_boundary(
        left_polyline: np.ndarray, right_polyline: np.ndarray
    ) -> np.ndarray:
        """
        Computes the width of a lanelet

        :param left_polyline: left boundary of lanelet
        :param right_polyline: right boundary of lanelet
        :return: width along lanelet
        """
        width_along_lanelet = np.zeros((len(left_polyline),))
        for i in range(len(left_polyline)):
            width_along_lanelet[i] = np.linalg.norm(
                left_polyline[i] - right_polyline[i]
            )
        return width_along_lanelet

    @staticmethod
    def create_curvilinear_coordinate_system_from_reference(
        ref_path: np.array, road_network_param: Dict
    ) -> CurvilinearCoordinateSystem:
        """
        Generates curvilinear coordinate system for a reference path

        :param ref_path: reference path (polyline)
        :param road_network_param: dictionary containing parameters of the road network
        :returns curvilinear coordinate system for reference path
        """
        new_ref_path = ref_path
        for i in range(0, road_network_param.get("num_chankins_corner_cutting")):
            new_ref_path = chaikins_corner_cutting(new_ref_path)
        new_ref_path = resample_polyline(
            new_ref_path, road_network_param.get("polyline_resampling_step")
        )

        curvilinear_cosy = CurvilinearCoordinateSystem(new_ref_path)

        return curvilinear_cosy


class RoadNetwork:
    """
    Representation of the complete road network of a CommonRoad scenario abstracted to lanes
    """

    def __init__(self, lanelet_network: LaneletNetwork, road_network_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param road_network_param: dictionary with parameters for the road network
        """
        self.lanelet_network = lanelet_network
        self.lanes = self._create_lanes(road_network_param)

    def _create_lanes(self, road_network_param: Dict) -> List[Lane]:
        """
        Creates lanes for road network

        :param road_network_param: dictionary with parameters for the road network
        """
        lanes = []
        lane_lanelets = []
        start_lanelets = []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.predecessor) == 0:
                start_lanelets.append(lanelet)
            else:
                predecessors = [
                    self.lanelet_network.find_lanelet_by_id(pred_id)
                    for pred_id in lanelet.predecessor
                ]
                for pred in predecessors:
                    if not lanelet.lanelet_type == pred.lanelet_type:
                        start_lanelets.append(lanelet)
        for lanelet in start_lanelets:
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.ACCESS_RAMP
            elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.EXIT_RAMP
            elif LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
                lanelet_type = LaneletType.MAIN_CARRIAGE_WAY
            else:
                lanelet_type = None
            (
                merged_lanelets,
                merge_jobs,
            ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet,
                self.lanelet_network,
                road_network_param.get("merging_length"),
            )
            if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
                merged_lanelets.append(lanelet)
                merge_jobs.append([lanelet.lanelet_id])
            for idx in range(len(merged_lanelets)):
                lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))
        for lane_element in lane_lanelets:
            lanes.append(Lane(lane_element[0], lane_element[1], road_network_param))

        return lanes

    def find_lane_ids_by_obstacle(self, obstacle_id: int, time_step: int) -> Set[int]:
        """
        Finds the lanes an obstacle belongs to and returns their IDs

        :param obstacle_id: ID of the obstacle
        :param time_step: time step of interest
        """
        lane_ids = set()
        for lane in self.lanes:
            if obstacle_id in lane.lanelet.dynamic_obstacle_by_time_step(time_step):
                lane_ids.add(lane.lanelet.lanelet_id)

        return lane_ids

    def find_lane_ids_by_lanelets(self, lanelets: Set[int]) -> Set[int]:
        """
        Finds the lanes given set of lanelets belong to and returns their IDs

        :param lanelets: list of lanelet IDs
        :returns set of lanelet IDs
        """
        lane_ids = set()
        for lane in self.lanes:
            for lanelet_id in lanelets:
                if lanelet_id in lane.contained_lanelets:
                    lane_ids.add(lane.lanelet.lanelet_id)

        return lane_ids

    def find_lanes_by_lanelets(self, lanelets: Set[int]) -> Set[Lane]:
        """
        Finds the lanes to which a given set of lanelets belongs to

        :param lanelets: list of lanelet IDs
        :returns set of lane objects
        """
        lanes = set()
        for lane in self.lanes:
            for lanelet_id in lanelets:
                if lanelet_id in lane.contained_lanelets:
                    lanes.add(lane)

        return lanes

    def find_lane_by_lanelet(self, lanelet_id: int) -> Lane:
        """
        Finds the lane a lanelet belongs to

        :param lanelet_id: CommonRoad lanelet ID
        :returns lane object
        """
        for lane in self.lanes:
            if lanelet_id in lane.contained_lanelets:
                return lane

    def find_lane_by_obstacle(
        self, obs_lanelet_center: List[int], obs_lanelet_shape: List[int]
    ) -> Lane:
        """
        Finds the lanes an obstacle occupies

        :param obs_lanelet_center: IDs of lanelet the obstacle center is on (use only first one)
        :param obs_lanelet_shape: IDs of lanelet the obstacle shape is on
        :returns lane the obstacle center is on
        """

        occupied_lanes = set()
        lanelets_center_updated = obs_lanelet_center
        obs_lanelet_shape_updated = obs_lanelet_shape
        if len(obs_lanelet_center) > 0:
            for lane in self.lanes:
                for lanelet in lanelets_center_updated:
                    if lanelet in lane.contained_lanelets:
                        occupied_lanes.add(lane)
        else:
            for lane in self.lanes:
                for lanelet in obs_lanelet_shape_updated:
                    if lanelet in lane.contained_lanelets:
                        occupied_lanes.add(lane)
        if len(occupied_lanes) == 1:
            return list(occupied_lanes)[0]
        for lane in occupied_lanes:
            for lanelet_id in lane.contained_lanelets:
                if (
                    LaneletType.MAIN_CARRIAGE_WAY
                    in self.lanelet_network.find_lanelet_by_id(lanelet_id).lanelet_type
                ):
                    return lane
        return list(occupied_lanes)[0]
