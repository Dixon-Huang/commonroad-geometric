from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Any
from omegaconf import OmegaConf
import numpy as np
import torch
from gymnasium.spaces import Box, Space
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace, \
    BaseControlSpaceOptions
from commonroad_geometric.common.config import Config
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_rp.utility.config import ReactivePlannerConfiguration, _dict_to_params
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad.scenario.state import InitialState
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner
from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.reactive_planner import ReactivePlanner
from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.cost_function.robustness_cost_function_simplified import RobustnessCostFunctionSimplified
# from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.cost_function.robustness_cost_function import RobustnessCostFunction
from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.cost_function.baseline_cost_function import BaselineCostFunction
from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.cost_function.utils.environment_extractor import extract_environment_info
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import \
    FlattenedGraphObserver

import logging
logger = logging.getLogger(__name__)

@dataclass
class ReactivePlannerActorOptions(BaseControlSpaceOptions):
    reactive_planner_options: Config


class ReactivePlannerActorSimplified:
    """
    Control space using reactive planner for generating trajectories.
    """
    config: ReactivePlannerConfiguration

    def __init__(
            self,
            options=None,
    ) -> None:
        config = options.reactive_planner_options
        config = OmegaConf.create(config.as_dict())
        self.config = _dict_to_params(config, ReactivePlannerConfiguration)
        self.uuid = None
        self.optimal_trajectory = None  # 缓存的最优轨迹
        self.current_count = 0
        self.planner: ReactivePlanner = None

        # 初始化环境信息
        self.environment_data = None
        self.environment_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化 observer
        self.observer = FlattenedGraphObserver(data_padding_size=200, global_features_include=None)

    def step(
            self,
            ego_vehicle_simulations: EgoVehicleSimulation,
            value_function,
    ) -> bool:

        try:
            ego_vehicle_simulation = ego_vehicle_simulations[0]
            new_uuid = ego_vehicle_simulation.uuid


            if self.uuid is None or new_uuid != self.uuid:
                self.current_count = 0
                self.uuid = new_uuid

                # 获取当前的场景和规划问题
                scenario: Scenario = ego_vehicle_simulation.current_scenario
                planning_problem: PlanningProblemSet = ego_vehicle_simulation.planning_problem
                # end_velocity = planning_problem.goal.state_list[0].velocity.end

                # 可视化最初场景和规划问题
                # from projects.graph_rl_agents.predictive_traffic_rule_compliance.run_scenario_visualization import visualize_scenario
                # visualize_scenario(scenario, planning_problem)

                # 更新配置中的场景和规划问题
                self.config.update(scenario=scenario, planning_problem=planning_problem)

                # *************************************
                # Initialize Planner
                # *************************************
                # run route planner and add reference path to config
                route_planner = RoutePlanner(self.config.scenario.lanelet_network, self.config.planning_problem)
                route = route_planner.plan_routes().retrieve_first_route()

                # 初始化 reactive_planner
                self.planner = ReactivePlanner(self.config)

                # set reference path for curvilinear coordinate system and desired velocity
                self.planner.set_reference_path(route.reference_path)

            plan_new_trajectory = self.current_count % self.config.planning.replanning_frequency == 0

            if plan_new_trajectory:

                # 提取环境信息
                if self.environment_data is None or self.environment_info is None: # 可能需要在场景变换的时候就更新
                    ego_vehicle_simulation_deepcopy = deepcopy(ego_vehicle_simulation) # 是否可以删除？
                    self.environment_data, self.environment_info = extract_environment_info(
                        ego_vehicle_simulation_deepcopy,
                        self.observer,
                        self.device
                    )

                    # 添加必要的信息到 environment_info
                    self.environment_info[
                        'ego_vehicle_params_b'] = ego_vehicle_simulation_deepcopy.ego_vehicle.parameters.b
                    self.environment_info['current_time_step'] = ego_vehicle_simulation_deepcopy.current_time_step

                # ego_vehicle_simulation_deepcopy = deepcopy(ego_vehicle_simulation)
                self.planner.set_desired_velocity(current_speed=self.planner.x_0.velocity)

                # set cost function
                self.planner.set_cost_function(RobustnessCostFunctionSimplified(
                    desired_speed=self.planner.desired_speed,
                    desired_d=0.0,
                    desired_s=self.planner.desired_lon_position,
                    environment_data=self.environment_data,
                    environment_info=self.environment_info,
                    value_function=value_function,
                    device=self.device
                ))

                # **************************
                # Run Planning
                # **************************
                # 使用 reactive_planner 生成轨迹
                optimal = self.planner.plan()

                if optimal is None:
                    x_0 = self.planner.x_0
                    ego_vehicle = self.planner.convert_state_list_to_commonroad_object([x_0])
                    self.visualize_trajectory(ego_vehicle, self.planner)
                    logger.warning("Planning failed. Requesting scenario reset.")
                    return None  # 表示需要重置场景

                self.optimal_trajectory = optimal  # 缓存生成的最优轨迹
                ego_vehicle = self.planner.convert_state_list_to_commonroad_object([self.optimal_trajectory[0].state_list[1]])

                self.visualize_trajectory(ego_vehicle, self.planner)

                self.current_count += 1

                # record state and input
                self.planner.record_state_and_input(self.optimal_trajectory[0].state_list[1])
                # reset planner state for re-planning
                self.planner.reset(initial_state_cart=self.planner.record_state_list[-1],
                                   initial_state_curv=(self.optimal_trajectory[2][1], self.optimal_trajectory[3][1]),
                                   collision_checker=self.planner.collision_checker,
                                   coordinate_system=self.planner.coordinate_system)

                logger.info(f"current count {self.current_count}")

                # orientation = self.optimal_trajectory[0].state_list[1].orientation
                # velocity = self.optimal_trajectory[0].state_list[1].velocity
                # array = [np.array([orientation, velocity])]

            else:
                # continue on optimal trajectory
                temp = self.current_count % self.config.planning.replanning_frequency
                # record state and input
                ego_vehicle = self.planner.convert_state_list_to_commonroad_object([self.optimal_trajectory[0].state_list[1+temp]])

                self.visualize_trajectory(ego_vehicle, self.planner)

                # record state and input
                self.planner.record_state_and_input(self.optimal_trajectory[0].state_list[1+temp])
                # reset planner state for re-planning
                self.planner.reset(initial_state_cart=self.planner.record_state_list[-1],
                                   initial_state_curv=(self.optimal_trajectory[2][1+temp], self.optimal_trajectory[3][1+temp]),
                                   collision_checker=self.planner.collision_checker,
                                   coordinate_system=self.planner.coordinate_system)

                self.current_count += 1

                logger.info(f"current count {self.current_count}")

                # orientation = self.optimal_trajectory[0].state_list[1+temp].orientation
                # velocity = self.optimal_trajectory[0].state_list[1+temp].velocity
                # array = [np.array([orientation, velocity])]

            # # Unpack tuple values
            position_x, position_y = ego_vehicle.prediction.trajectory.final_state.position
            velocity = ego_vehicle.prediction.trajectory.final_state.velocity
            orientation = ego_vehicle.prediction.trajectory.final_state.orientation
            yaw_rate = ego_vehicle.prediction.trajectory.final_state.yaw_rate
            acceleration = ego_vehicle.prediction.trajectory.final_state.acceleration
            time_step = ego_vehicle.prediction.trajectory.final_state.time_step
            slip_angle = 0

            # Create numpy array with unpacked values
            array = [np.array([position_x, position_y, velocity, orientation, yaw_rate, acceleration, time_step, slip_angle])]

            return array

        except ValueError as e:
            if "Initial state could not be transformed" in str(e):
                logger.error("Initial state transformation failed. Requesting scenario reset.")
                return None  # 表示需要重置场景
            else:
                logger.error(f"Error in reactive planner actor: {e}")
                return None

    def visualize_trajectory(self, ego_vehicle, reactive_planner):
        if self.config.debug.show_plots or self.config.debug.save_plots:
            sampled_trajectory_bundle = None
            if self.config.debug.draw_traj_set:
                sampled_trajectory_bundle = deepcopy(reactive_planner.stored_trajectories)

        if self.config.debug.show_plots or self.config.debug.save_plots:
            visualize_planner_at_timestep(scenario=self.config.scenario,
                                          planning_problem=self.config.planning_problem,
                                          ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                          ref_path=reactive_planner.reference_path,
                                          timestep=ego_vehicle.prediction.trajectory.final_state.time_step,
                                          config=self.config)