from abc import ABC
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import copy
import math
import concurrent.futures
from commonroad_rp.cost_function import CostFunction
from commonroad.scenario.state import State, InitialState, CustomState, InputState
from stable_baselines3.common.utils import obs_as_tensor
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver
import commonroad_rp.trajectories

logger = logging.getLogger(__name__)

class RobustnessCostFunctionSimplified(CostFunction):
    """
    Cost function integrating value function as the robustness measure.
    """

    def __init__(
        self,
        desired_speed: Optional[float] = None,
        desired_d: float = 0.0,
        desired_s: Optional[float] = None,
        environment_data_dict: Any = None,
        environment_info: Dict[str, Any] = None,
        value_function: Any = None,
        device: torch.device = None,
    ):
        super(RobustnessCostFunctionSimplified, self).__init__()
        # Target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s
        self.value_function = value_function
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observer = FlattenedGraphObserver(data_padding_size=200, global_features_include=None)
        self.environment_data_dict = environment_data_dict
        self.environment_info = environment_info

        # Setup observer with dummy data
        self._observation_space = self.observer.setup(
            dummy_data=self.environment_data
        )

        # Weights
        self.w_a = 5  # Acceleration weight
        self.w_r = 5  # Robustness weight

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        try:
            costs = 0.0

            # Evaluate the trajectory
            value_list = self._calculate_robustness(trajectory)
            value_array = np.vstack([v.detach().cpu().numpy() for v in value_list])
            costs -= 1e4 * (
                np.sum(5 * value_array) +
                (50 * value_array[-1]) +
                (100 * value_array[int(len(value_array) / 2) - 1])
            )

            # Other cost calculations
            # Acceleration costs
            costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)
            # Velocity costs
            if self.desired_speed is not None:
                costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                         (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                         (100 * (trajectory.cartesian.v[int(len(trajectory.cartesian.v) / 2)] - self.desired_speed) ** 2)
            if self.desired_s is not None:
                costs += np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                         (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2
            # Distance costs
            costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                     (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
            # Orientation costs
            costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + \
                     (5 * np.abs(trajectory.curvilinear.theta[-1])) ** 2

        except Exception as e:
            logger.error(f"Error in cost function calculation: {e}")

        return float(costs)

    def _calculate_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        """
        Calculate the robustness value for the given trajectory.
        """
        value_list = []
        num_points = len(trajectory.cartesian.x)
        indices = self._get_evaluation_indices(num_points)
        indices_set = set(indices)  # For faster lookup

        for i in range(len(trajectory.cartesian.x)):
            # Transform coordinates
            position_rear = np.array([trajectory.cartesian.x[i], trajectory.cartesian.y[i]])
            orientation = trajectory.cartesian.theta[i]
            position_center = position_rear + self.environment_info['ego_vehicle_params_b'] * np.array(
                [math.cos(orientation), math.sin(orientation)]
            )

            state = State(
                position=position_center,
                velocity=trajectory.cartesian.v[i],
                acceleration=trajectory.cartesian.a[i],
                orientation=orientation,
                time_step=self.environment_info['current_time_step'] + i + 1,  # Adjust time step as needed
                yaw_rate=0.0,
                slip_angle=0.0
            )

            if i in indices_set:
                # Extract obs and compute value at specified time steps
                time_step = state.time_step
                obs_tensor = self._extract_obs_at_state(
                    state,
                    self.environment_data_dict,
                    time_step,
                    self.environment_info,
                    self.observer,
                    self.device
                )
                with torch.no_grad():
                    value = self.value_function.predict_values(obs_tensor)
                value_list.append(value)
        return value_list

    def _extract_obs_at_state(
        self,
        ego_state,
        environment_data,
        environment_info,
        observer,
        device
    ):
        # Create a new data object, include the current ego vehicle state
        data = copy.deepcopy(environment_data)
        data.ego_vehicle_state = ego_state

        # Use observer to extract obs
        obs = observer.observe(
            data=data,
            ego_vehicle_simulation=None  # Not needed
        )
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, device)
        return obs_tensor

    def _get_evaluation_indices(self, num_points):
        """
        Get the indices corresponding to the specific points where robustness is evaluated.
        """
        idx_mid = int(round(0.5 * (num_points - 1)))
        idx_end = num_points - 1

        indices = [idx_mid, idx_end]
        # Remove duplicates and sort
        indices = sorted(set(indices))
        return indices

    @staticmethod
    def compute_costs_parallel(
            trajectories: List[commonroad_rp.trajectories.TrajectorySample],
            cost_function: 'RobustnessCostFunctionSimplified',
            max_workers: Optional[int] = None
    ) -> List[float]:
        """
        Compute costs for a list of trajectories in parallel.

        Args:
            trajectories: List of trajectory samples.
            cost_function: An instance of RobustnessCostFunctionSimplified.
            max_workers: Maximum number of worker processes.

        Returns:
            List of costs corresponding to the trajectories.
        """
        costs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare data that can be serialized
            serialized_environment_data_dict = cost_function.environment_data_dict
            serialized_environment_info = cost_function.environment_info
            value_function_state = cost_function.value_function.state_dict()
            device = cost_function.device
            observer_state = cost_function.observer.__dict__

            futures = [
                executor.submit(
                    _compute_trajectory_cost_parallel,
                    traj,
                    serialized_environment_data_dict,
                    serialized_environment_info,
                    value_function_state,
                    observer_state,
                    device,
                    cost_function.desired_speed,
                    cost_function.desired_d,
                    cost_function.desired_s
                )
                for traj in trajectories
            ]
            for future in concurrent.futures.as_completed(futures):
                costs.append(future.result())
        return costs


def _compute_trajectory_cost_parallel(
    trajectory,
    environment_data,
    environment_info,
    value_function_state,
    observer_state,
    device,
    desired_speed,
    desired_d,
    desired_s,
):
    """
    Helper function to compute the cost of a trajectory in a separate process.

    This function is necessary because some objects like value_function and observer
    cannot be pickled directly. Instead, we pass their states and recreate them in the child process.
    """
    # Recreate value_function and observer in the child process
    # Note: You need to replace 'YourValueFunctionClass' with the actual class of your value function
    value_function = YourValueFunctionClass()  # Replace with your actual class
    value_function.load_state_dict(value_function_state)
    value_function.to(device)
    value_function.eval()

    observer = FlattenedGraphObserver(data_padding_size=200, global_features_include=None)
    observer.__dict__.update(observer_state)

    # Recreate the cost function without complex classes
    cost_function = RobustnessCostFunction(
        desired_speed=desired_speed,
        desired_d=desired_d,
        desired_s=desired_s,
        environment_data=environment_data,
        environment_info=environment_info,
        value_function=value_function,
        device=device
    )

    # Compute the cost
    cost = cost_function.evaluate(trajectory)
    return cost
