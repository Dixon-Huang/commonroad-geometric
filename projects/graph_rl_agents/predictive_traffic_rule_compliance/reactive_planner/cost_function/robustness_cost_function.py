from abc import ABC, abstractmethod
import logging
from typing import Optional
from commonroad_rp.cost_function import CostFunction, DefaultCostFunction
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import \
    FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.commonroad_gym_env import CommonRoadGymEnv
from commonroad.scenario.state import State, InitialState, CustomState, InputState
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import copy
import torch
import math
import commonroad_rp.trajectories

logger = logging.getLogger(__name__)

class RobustnessCostFunction(CostFunction):
    """
    Cost function integrating value function as the robustness measure
    """

    def __init__(self, desired_speed: Optional[float] = None,
                 desired_d: float = 0.0,
                 desired_s: Optional[float] = None,
                 ego_vehicle_simulation: EgoVehicleSimulation = None,
                 value_function: object = None,
                 ) -> object:
        super(RobustnessCostFunction, self).__init__()
        # target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s
        self.value_function = value_function
        # self.ego_vehicle_simulation = copy.deepcopy(ego_vehicle_simulation)
        self.ego_vehicle_simulation = ego_vehicle_simulation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observer = FlattenedGraphObserver(data_padding_size=200, global_features_include=None)
        self._observation_space = self.observer.setup(
            dummy_data=self.ego_vehicle_simulation.extract_data()
        )

        # weights
        self.w_a = 5  # acceleration weight
        self.w_r = 5  # robustness weight

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        try:
            costs = 0.0
            # ego_vehicle_simulation = copy.deepcopy(self.ego_vehicle_simulation)
            saved_state = self.ego_vehicle_simulation.save_state()
            ego_vehicle_simulation = self.ego_vehicle_simulation

            # 评估轨迹
            # Replace future trajectory and extract scene features
            value_list = self._calculate_last_robustness(trajectory, ego_vehicle_simulation)
            value_array = np.vstack([v.detach().cpu().numpy() for v in value_list])
            costs -= self.w_r * 1e5 * value_array

            value_list = self._calculate_robustness(trajectory, ego_vehicle_simulation)
            value_array = np.vstack([v.detach().cpu().numpy() for v in value_list])
            costs -= 1e4 * (np.sum(5 * value_array) + \
                            (50 * value_array[-1]) + \
                            (100 * value_array[int(len(value_array) / 2) - 1]))

            # Other cost calculations
            # acceleration costs
            costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)
            # velocity costs
            if self.desired_speed is not None:
                costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                         (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                         (100 * (trajectory.cartesian.v[
                                     int(len(trajectory.cartesian.v) / 2)] - self.desired_speed) ** 2)
            if self.desired_s is not None:
                costs += np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                         (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2
            # distance costs
            costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                     (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
            # orientation costs
            costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                    5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        except Exception as e:
            logger.error(f"Error in cost function calculation: {e}")

        finally:
            # 无论评估是否成功，都恢复到原始状态
            self.ego_vehicle_simulation.restore_state(saved_state)
            pass

        return float(costs)

    def _calculate_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample,
                                   ego_vehicle_simulation: EgoVehicleSimulation):
        """
        Replace the future trajectory of the ego vehicle with the provided trajectory.
        """
        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        value_list = []
        num_points = len(trajectory.cartesian.x)
        indices = self._get_evaluation_indices(num_points)
        indices_set = set(indices)  # For faster lookup

        for i in range(len(trajectory.cartesian.x) - 1):
            # 转换坐标系
            position_rear = np.array([trajectory.cartesian.x[i + 1], trajectory.cartesian.y[i + 1]])
            orientation = trajectory.cartesian.theta[i + 1]
            position_center = position_rear + ego_vehicle.parameters.b * np.array(
                [math.cos(orientation), math.sin(orientation)])

            state = InitialState(
                position=position_center,
                velocity=trajectory.cartesian.v[i + 1],
                acceleration=trajectory.cartesian.a[i + 1],
                orientation=orientation,
                time_step=ego_vehicle_simulation.current_time_step + 1,
                yaw_rate=0.0,
                slip_angle=0.0
            )

            ego_vehicle.set_next_state(state)
            next(ego_vehicle_simulation.simulation)

            if i + 1 in indices_set:
                obs_tensor = self._extract_scene_features(ego_vehicle_simulation)
                with torch.no_grad():
                    value = self.value_function.predict_values(obs_tensor)
                value_list.append(value)
        return value_list

    def _calculate_last_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample,
                                   ego_vehicle_simulation: EgoVehicleSimulation):
        """
        Replace the future trajectory of the ego vehicle with the provided trajectory.
        """
        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        value_list = []
        for i in range(len(trajectory.cartesian.x) - 1):
            # 转换坐标系
            position_rear = np.array([trajectory.cartesian.x[i + 1], trajectory.cartesian.y[i + 1]])
            orientation = trajectory.cartesian.theta[i + 1]
            position_center = position_rear + ego_vehicle.parameters.b * np.array(
                [math.cos(orientation), math.sin(orientation)])

            state = InitialState(
                position=position_center,
                velocity=trajectory.cartesian.v[i + 1],
                acceleration=trajectory.cartesian.a[i + 1],
                orientation=orientation,
                time_step=ego_vehicle_simulation.current_time_step + 1,
                yaw_rate=0.0,
                slip_angle=0.0
            )

            ego_vehicle.set_next_state(state)
            next(ego_vehicle_simulation.simulation)
        obs_tensor = self._extract_scene_features(ego_vehicle_simulation)
        with torch.no_grad():
            value = self.value_function.predict_values(obs_tensor)
        value_list.append(value)
        return value_list

    def _extract_scene_features(self, ego_vehicle_simulation: EgoVehicleSimulation):
        """
        Extract the features of the current scene using the ego vehicle simulation.
        """
        # ego_vehicle_simulation.current_time_step
        data = ego_vehicle_simulation.extract_data()
        observer = self.observer
        obs = observer.observe(
            data=data,
            ego_vehicle_simulation=ego_vehicle_simulation
        )
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, self.device)
        return obs_tensor

    def _get_evaluation_indices(self, num_points):
        """
        Get the indices corresponding to the specific points where robustness is evaluated.
        """
        idx_q1 = int(round(0.25 * (num_points - 1)))
        idx_mid = int(round(0.5 * (num_points - 1)))
        idx_q3 = int(round(0.75 * (num_points - 1)))
        idx_end = num_points - 1

        indices = [
            # idx_q1,
            idx_mid,
            # idx_q3,
            idx_end
                   ]
        # Remove duplicates and sort
        indices = sorted(set(indices))
        return indices