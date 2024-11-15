from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, sin
from typing import Optional, TYPE_CHECKING

import numpy as np
from commonroad.scenario.trajectory import State
from gymnasium.spaces import Box, Space

from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, BaseControlSpaceOptions
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.utils.path_observer import PathObserver
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.utils.pid_controller import PIDController
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import ActionBase
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationFinishedException

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class DeterministicLateralControlMethod(Enum):
    FIXED_PATH = 'fixed path'
    PURE_PURSUIT = 'pure pursuit' # https://github.com/AtsushiSakai/PythonRobotics/blob/46b6981f34dd783132ea5a23b46af69dc6888024/PathTracking/pure_pursuit/pure_pursuit.py
    STANLEY = 'stanley' # https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf


@dataclass
class LongitudinalControlOptions(BaseControlSpaceOptions):
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    max_velocity: Optional[float] = None
    min_velocity: Optional[float] = 0.0
    fallback_to_navigator_route: bool = True
    pid_control: bool = False
    k_P_velocity: float = 3.0
    k_D_velocity: float = 0.2
    k_I_velocity: float = 0.0
    neg_acc_multiplier: float = 1.0 
    steering_method: DeterministicLateralControlMethod = DeterministicLateralControlMethod.STANLEY
    look_ahead_distance: float = 1.0 # [m]
    k_e: float = 12.0
    k_v: float = 8.0
    k_dy: float = 1.4
    k_ds: float = 1.35
    k_ss: float = 0.075
    gain: float = 1.0
    steering_bound: float = 0.4


class LongitudinalControlSpace(BaseControlSpace):
    """
    Low-level control space where the agent is tasked with 
    one-dimensional acceleration control along a predefined reference path.
    """

    def __init__(
        self,
        options: Optional[LongitudinalControlOptions] = None
    ) -> None:
        options = options or LongitudinalControlOptions()
        self.options = options

        if self.options.pid_control:
            self._pid_controller_velocity = PIDController(
                k_P=options.k_P_velocity,
                k_D=options.k_D_velocity,
                k_I=options.k_I_velocity
            )
        
        self.path_observer = PathObserver(
            look_ahead_distance=options.look_ahead_distance
        )

        super().__init__(options)

    @property
    def gym_action_space(self) -> Space:
        return Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype="float64"
        )

    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:
        longitudinal_action = action[-1]
        #longitudinal_action = 1.0 # TODO for debugging
        #print("WARN DISABLE THIS")
        ego_vehicle = ego_vehicle_simulation.ego_vehicle
        current_velocity = ego_vehicle.state.velocity

        lower_bound = self.options.lower_bound if self.options.lower_bound is not None else -ego_vehicle.parameters.longitudinal.a_max
        upper_bound = self.options.upper_bound if self.options.upper_bound is not None else ego_vehicle.parameters.longitudinal.a_max

        if self.options.pid_control:
            min_velocity = self.options.min_velocity if self.options.min_velocity is not None else 0.0
            max_velocity = self.options.max_velocity if self.options.max_velocity is not None else 20.0
            desired_velocity = min_velocity + (max_velocity - min_velocity) * (longitudinal_action + 1)/2
            error_velocity = desired_velocity - current_velocity
            acceleration = self._pid_controller_velocity(error_velocity, dt=ego_vehicle_simulation.dt)
        else:
            if longitudinal_action < 0:
                acceleration = -lower_bound * longitudinal_action
            else:
                acceleration = upper_bound * longitudinal_action

        if abs(acceleration) < 0:
            acceleration = acceleration * self.options.neg_acc_multiplier

        acceleration = np.clip(acceleration, lower_bound, upper_bound)

        next_velocity = current_velocity + acceleration * ego_vehicle_simulation.dt
        min_velocity = current_velocity + acceleration * ego_vehicle_simulation.dt

        if self.options.max_velocity is not None and next_velocity >= self.options.max_velocity:
            acceleration = (self.options.max_velocity - current_velocity) / ego_vehicle_simulation.dt

        if self.options.min_velocity is not None and next_velocity <= self.options.min_velocity:
            acceleration = (self.options.min_velocity - current_velocity) / ego_vehicle_simulation.dt

        if self.options.steering_method == DeterministicLateralControlMethod.FIXED_PATH:
            next_state = self._interpolate_next_state(
                current_state=ego_vehicle.state,
                next_acceleration=acceleration,
                dt=ego_vehicle_simulation.simulation.dt
            )
            ego_vehicle.set_next_state(next_state)
        elif self.options.steering_method in {DeterministicLateralControlMethod.PURE_PURSUIT, DeterministicLateralControlMethod.STANLEY}:
            steering_angle = self._get_steering_angle(
                ego_vehicle_simulation=ego_vehicle_simulation,
                current_state=ego_vehicle.state
            )
            if not np.isfinite(steering_angle):
                steering_angle = 0.0
            control_action = np.array([
                steering_angle,
                acceleration
            ], dtype=np.float64)
            ego_vehicle.update_current_state(
                action=control_action,
                action_base=ActionBase.ACCELERATION
            )
        else:
            raise NotImplementedError(f"{self.options.steering_method=}")

        return True

    def _get_steering_angle(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        current_state: State
    ) -> float:
        ego_parameters = ego_vehicle_simulation.ego_vehicle.parameters
        obs = self.path_observer.observe(
            ego_vehicle_simulation=ego_vehicle_simulation
        )

        if self.options.steering_method == DeterministicLateralControlMethod.PURE_PURSUIT:
            steering_angle = atan2(
                2.0 * ego_parameters.l * sin(obs.yaw_diff_look_ahead) / self.options.look_ahead_distance,
                1.0
            )
        else:
            # Stanley
            yaw_diff_crosstrack = np.arctan(
                self.options.k_e * obs.crosstrack_error_look_ahead / (self.options.k_v + current_state.velocity)
            )
            #yaw_diff_correction = self.options.k_dy * (- yaw_rate_front - yaw_rate_path_front)
            yaw_diff_correction = self.options.k_dy * (- obs.yaw_rate_front - obs.yaw_rate_path_front)

            psi_ss = self.options.k_ss * current_state.velocity * obs.yaw_rate_path_look_ahead

            steering_angle =  - obs.yaw_diff_look_ahead + psi_ss + yaw_diff_correction + yaw_diff_crosstrack

        steering_angle *= self.options.gain

        steering_angle_diff = steering_angle - current_state.steering_angle

        d_steer_correction = self.options.k_ds * steering_angle_diff
        steering_angle += d_steer_correction

        steering_angle_clipped = max(
            min(
                steering_angle,
                self.options.steering_bound
            ), -self.options.steering_bound
        )

        # print(f"{current_state.steering_angle:+.3f}: {yaw_diff_look_ahead=:+.3f} {yaw_rate_look_ahead=:+.3f} | {psi_ss=:+.3f} | CS: {steering_angle_diff:+.3f} -> {d_steer_correction:+.3f} - CY: {yaw_rate_diff_look_ahead:+.3f} - {yaw_rate_path_look_ahead:+.3f} -> {yaw_diff_correction:+.3f} || ct: {yaw_diff_crosstrack:+.3f} | {steering_angle_clipped:.4f}")

        return steering_angle_clipped

    def _interpolate_next_state(
        self,
        current_state: State,
        next_acceleration: float,
        dt: float
    ) -> State:

        assert  self._trajectory_polyline is not None

        if self._last_arclength is None:
            arclength = self._trajectory_polyline.get_projected_arclength(current_state.position)
        else:
            arclength = self._last_arclength
        next_velocity = current_state.velocity + next_acceleration * dt
        next_arclength = arclength + next_velocity * dt
        self._last_arclength = next_arclength
        next_position = self._trajectory_polyline(arclength=next_arclength)
        # Next state would be outside of trajectory polyline -> we either finished following the trajectory or left it due to other circumstances
        if np.isnan(next_position).any():
            raise EgoVehicleSimulationFinishedException()
        next_orientation = self._trajectory_polyline.get_direction(arclength=next_arclength, use_cached=False)
        if np.isnan(next_orientation):
            raise RuntimeError("New error, next_orientation would be NaN")

        return State(
            position=next_position,
            orientation=next_orientation,
            velocity=next_velocity,
            acceleration=next_acceleration,
            time_step=current_state.time_step + 1
        )

    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        if self.options.pid_control:
            self._pid_controller_velocity.clear()

        self.path_observer.reset()

        assert ego_vehicle_simulation.ego_vehicle.ego_route is not None
        ego_vehicle_id = -1
        ego_planning_problem = ego_vehicle_simulation.planning_problem
        if ego_planning_problem is None or ego_planning_problem.goal is None:
            raise RuntimeError(f"Longitudinal control space requires planning problem, missing planning problem.")

        # Ego vehicle obstacle has to already be removed from current scenario as the accelerations/states will be different from the recorded trajectory
        # Recorded trajectory can be accessed through the initial scenario
        initial_ego_vehicle_obstacle = ego_vehicle_simulation.simulation.initial_scenario._dynamic_obstacles.get(ego_vehicle_id, None)
        if initial_ego_vehicle_obstacle is None:
            if not self.options.fallback_to_navigator_route:
                raise RuntimeError(f"Longitudinal control space requires trajectory, missing trajectory for ego vehicle with ID {ego_vehicle_id} in initial scenario of simulation.")
            self._trajectory_polyline = ego_vehicle_simulation.ego_route.planning_problem_path_polyline
        else:
            obstacle_initial_time_step = initial_ego_vehicle_obstacle.initial_state.time_step
            obstacle_final_time_step = initial_ego_vehicle_obstacle.prediction.final_time_step
            time_step_delta = obstacle_final_time_step - obstacle_initial_time_step + 1
            if time_step_delta <= 1:
                if not self.options.fallback_to_navigator_route:
                    raise RuntimeError(f"Longitudinal control space requires trajectory, missing trajectory for ego vehicle with ID {ego_vehicle_id} in initial scenario of simulation.")
                self._trajectory_polyline = ego_vehicle_simulation.ego_route.planning_problem_path_polyline
            else:
                num_states = len(initial_ego_vehicle_obstacle.prediction.trajectory.state_list)
                has_faulty_trajectory = num_states != time_step_delta
                if has_faulty_trajectory:
                    raise RuntimeError(f"Longitudinal control space requires full trajectory, missing trajectory states, len(state_list)={num_states}, expected={time_step_delta}")

                positions = np.array([state_at_time(initial_ego_vehicle_obstacle, t, assume_valid=True).position
                                    for t in range(obstacle_initial_time_step, obstacle_final_time_step + 1)])
                ego_vehicle_simulation.ego_vehicle.ego_route.planning_problem_path = positions
                self._trajectory_polyline = ego_vehicle_simulation.ego_route.planning_problem_path_polyline
