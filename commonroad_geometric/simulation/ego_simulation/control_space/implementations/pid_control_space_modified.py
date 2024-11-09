from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Space

from commonroad_geometric.common.geometry.helpers import relative_orientation, make_valid_orientation
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.utils.pid_controller import PIDController
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad.scenario.state import InitialState

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def shortest_angle_diff(target_angle, current_angle):
    """Calculate the shortest difference between two angles, considering wrap-around."""
    normalized_diff = normalize_angle(target_angle - current_angle)
    return normalized_diff

def calculate_circular_difference(angle1, angle2):
    """Calculate the shortest circular difference between two angles."""
    diff = normalize_angle(angle1 - angle2)
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return diff

def wrap_orientation(orientation):
    """Wrap the orientation to stay within [-pi, pi] range, handling boundary crossing explicitly."""
    if orientation > np.pi:
        return orientation - 2 * np.pi
    elif orientation < -np.pi:
        return orientation + 2 * np.pi
    return orientation

@dataclass
class PIDControlOptions(BaseControlSpaceOptions):
    lower_bound_acceleration: float = -10.0
    upper_bound_acceleration: float = 6.5
    lower_bound_velocity: float = 1e-3
    upper_bound_velocity: Optional[float] = None
    lower_bound_steering: float = -0.4
    upper_bound_steering: float = 0.4

    min_velocity_steering: float = 1.0
    k_P_orientation: float = 6.0  # increase to increase aggresiveness
    k_D_orientation: float = 0.0  # increase to counter overshooting behavior
    k_I_orientation: float = 0.0  # increase to counter stationary error
    k_yaw_rate: float = 1.75

    windup_guard_orientation: float = 10.0
    k_P_velocity: float = 7.5
    k_D_velocity: float = 0.5
    k_I_velocity: float = 0.0

    use_lanelet_coordinate_frame: bool = False
    steering_coefficient: float = 1.5
    steering_error_clipping: float = np.pi/6


class PIDControlSpace(BaseControlSpace):
    """
    Low-level space where the contol actions correspond to
    setting the reference setpoints for longitudinal and lateral PID controllers.
    """

    def __init__(
        self,
        options: Optional[PIDControlOptions] = None
    ) -> None:
        options = options or PIDControlOptions()
        self._options = options
        self._lower_bound_acceleration = options.lower_bound_acceleration
        self._upper_bound_acceleration = options.upper_bound_acceleration
        self._lower_bound_velocity = options.lower_bound_velocity
        self._upper_bound_velocity = options.upper_bound_velocity

        self._upper_bound_steering = options.upper_bound_steering
        self._lower_bound_steering = options.lower_bound_steering
        self._use_lanelet_coordinate_frame = options.use_lanelet_coordinate_frame
        self._last_lanelet_orientation = 0.0
        self._desired_orientation = 0.0

        self._pid_controller_orientation = PIDController(k_P=options.k_P_orientation,
                                                         k_D=options.k_D_orientation,
                                                         k_I=options.k_I_orientation,
                                                         windup_guard=options.windup_guard_orientation,
                                                         d_threshold=2.0)
        self._pid_controller_velocity = PIDController(k_P=options.k_P_velocity,
                                                      k_D=options.k_D_velocity,
                                                      k_I=options.k_I_velocity)

        super().__init__(options)

    @property
    def gym_action_space(self) -> Space:
        return Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype="float64"
        )

    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:

        # 创建 position 为 numpy 数组
        position = np.array([float(action[0]), float(action[1])])
        # 使用 ego_vehicle_simulation 的时间步长，而不是从 action 中获取
        current_time_step = ego_vehicle_simulation.current_time_step + 1

        next_state = InitialState(
            position=position,
            velocity=float(action[2]),
            orientation=float(action[3]),
            yaw_rate=float(action[4]),
            acceleration=float(action[5]),
            # time_step=int(action[6]),  # 增加时间步长
            time_step=current_time_step,  # 使用 ego_vehicle_simulation 的时间步长
            slip_angle=float(action[7]) if len(action) > 6 else 0.0  # 如果提供了滑移角，则使用它
        )

        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        ego_vehicle_simulation.ego_vehicle.set_next_state(next_state)
        return True

    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        self._last_lanelet_orientation = 0.0
        self._pid_controller_orientation.clear()
        self._pid_controller_velocity.clear()
