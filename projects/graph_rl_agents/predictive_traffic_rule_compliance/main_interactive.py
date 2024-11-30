import matplotlib.pyplot as plt
import sys
import os
import copy
import hydra
import logging
import numpy as np
from torch import nn
from torch.optim import Adam
from math import sin, cos
from dataclasses import dataclass
from omegaconf import OmegaConf
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_rp.utility.logger import initialize_logger
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.cost_function import CostFunction, DefaultCostFunction, DefaultCostFunctionFailSafe
from commonroad_rp.utility.config import ReactivePlannerConfiguration, _dict_to_params
from commonroad_geometric.learning.reinforcement.project.hydra_rl_config import RLProjectConfig
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, \
    TrafficFeatureComputerOptions
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import \
    FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.pid_control_space_modified import \
    PIDControlOptions, PIDControlSpace
# from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle, VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.planning_problem import EgoRoute
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import (RandomRespawner,
                                                                                       RandomRespawnerOptions)
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner
from projects.graph_rl_agents.predictive_traffic_rule_compliance.reactive_planner.reactive_planner import \
    ReactivePlanner
# from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_geometric.common.config import Config
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.state import ReactivePlannerState
from projects.geometric_models.drivable_area.project import create_lanelet_graph_conversion_steps
from projects.graph_rl_agents.v2v_policy.feature_extractor import VehicleGraphFeatureExtractor
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.actor.reactive_planner_actor import \
    ReactivePlannerActor, ReactivePlannerActorOptions
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.reactive_planner_agent import \
    ReactivePlannerAgent
from projects.graph_rl_agents.predictive_traffic_rule_compliance.project import (V_FEATURE_COMPUTERS,
                                                                                 V2V_FEATURE_COMPUTERS,
                                                                                 L_FEATURE_COMPUTERS,
                                                                                 L2L_FEATURE_COMPUTERS,
                                                                                 V2L_FEATURE_COMPUTERS,
                                                                                 create_rewarders,
                                                                                 create_termination_criteria,
                                                                                 create_scenario_filterers,
                                                                                 create_scenario_preprocessors,
                                                                                 RENDERER_OPTIONS)
from projects.graph_rl_agents.predictive_traffic_rule_compliance.config.custom_hydra_rl_config import \
    CustomRLProjectConfig
from projects.graph_rl_agents.predictive_traffic_rule_compliance.evaluation.baseline_cost_function_for_evaluation import \
    BaselineCostFunction
from projects.graph_rl_agents.predictive_traffic_rule_compliance.evaluation.safe_brake_cost_function import \
    SafeBrakeCostFunction
from projects.graph_rl_agents.predictive_traffic_rule_compliance.evaluation.collision_avoiding_cost_function_for_evaluation import \
    CollisionAvoidingCostFunction
from projects.graph_rl_agents.predictive_traffic_rule_compliance.run_scenario_visualization import \
    visualize_scenario
from projects.graph_rl_agents.predictive_traffic_rule_compliance.change_planner import check_state
from projects.graph_rl_agents.predictive_traffic_rule_compliance.level_k_planner import level_k_planner
from projects.graph_rl_agents.predictive_traffic_rule_compliance.prediction.predict_constant_velocity import \
    predict_constant_velocity
from projects.graph_rl_agents.predictive_traffic_rule_compliance.lattice.Lattice_CRv3 import Lattice_CRv3
from projects.graph_rl_agents.predictive_traffic_rule_compliance.lattice.MCTs_CR import MCTs_CR
from projects.graph_rl_agents.predictive_traffic_rule_compliance.lattice.intersection_planner import \
    front_vehicle_info_extraction, IntersectionPlanner

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
from commonroad_dc.feasibility.feasibility_checker import state_transition_feasibility
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, VehicleParameterMapping
# from commonroad_dc.costs.evaluation import CostFunctionEvaluator
from commonroad_dc.feasibility.solution_checker import valid_solution
from commonroad.scenario.trajectory import State, Trajectory, CustomState
from sumocr.visualization.video import create_video
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.interface.sumo_simulation import SumoSimulation
from projects.graph_rl_agents.predictive_traffic_rule_compliance.sumo_simulation.utility import save_solution
from projects.graph_rl_agents.predictive_traffic_rule_compliance.sumo_simulation.simulations import \
    load_sumo_configuration
from projects.graph_rl_agents.predictive_traffic_rule_compliance.k_level.utility import distance_lanelet, brake, \
    Ipaction
from commonroad.common.solution import PlanningProblemSolution, Solution, CommonRoadSolutionWriter, VehicleType, \
    VehicleModel, CostFunction

from crmonitor.common.world import World
from crmonitor.common.vehicle import ControlledVehicle

sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def configure_experiment(cfg: dict) -> RLExperimentConfig:
    experiment_config = RLExperimentConfig(
        simulation_cls=ScenarioSimulation if cfg["enable_traffic"] else UnpopulatedSimulation,
        simulation_options=ScenarioSimulationOptions(
            lanelet_assignment_order=LaneletAssignmentStrategy.ONLY_SHAPE,
            lanelet_graph_conversion_steps=create_lanelet_graph_conversion_steps(
                enable_waypoint_resampling=cfg["enable_waypoint_resampling"],
                waypoint_density=cfg["lanelet_waypoint_density"]
            ),
            linear_lanelet_projection=cfg["linear_lanelet_projection"]
        ),

        control_space_cls=PIDControlSpace,
        control_space_options=PIDControlOptions(),
        respawner_cls=RandomRespawner,
        respawner_options=RandomRespawnerOptions(
            random_init_arclength=True,
            random_goal_arclength=True,
            random_start_timestep=True,
            only_intersections=False,
            route_length=(10, 15),
            init_speed=10.0,
            min_goal_distance=None,
            min_goal_distance_l2=cfg["spawning"]["min_goal_distance"],
            max_goal_distance_l2=cfg["spawning"]["max_goal_distance"],
            max_goal_distance=None,
            min_remaining_distance=None,
            max_attempts_outer=50,
            min_vehicle_distance=cfg["spawning"]["min_vehicle_distance"],
            min_vehicle_speed=None,
            min_vehicles_route=None,
            max_attempts_inner=5
        ),
        traffic_extraction_options=TrafficExtractorOptions(
            edge_drawer=VoronoiEdgeDrawer(
                dist_threshold=cfg["dist_threshold_v2v"]
            ) if cfg["edge_drawer_class_name"] != "KNearestEdgeDrawer" else KNearestEdgeDrawer(
                k=cfg["edge_drawer_k"],
                dist_threshold=cfg["dist_threshold_v2v"],
            ),
            feature_computers=TrafficFeatureComputerOptions(
                v=V_FEATURE_COMPUTERS,
                v2v=V2V_FEATURE_COMPUTERS,
                l=L_FEATURE_COMPUTERS,
                l2l=L2L_FEATURE_COMPUTERS,
                v2l=V2L_FEATURE_COMPUTERS,
            ),
            postprocessors=[],
            only_ego_inc_edges=False,
            assign_multiple_lanelets=True,
            ego_map_radius=cfg["ego_map_radius"]
        ),
        ego_vehicle_simulation_options=EgoVehicleSimulationOptions(
            vehicle_model=VehicleModel.KS,
            vehicle_type=VehicleType.FORD_ESCORT
        ),
        rewarder=SumRewardAggregator(create_rewarders(1)),
        termination_criteria=create_termination_criteria(),
        env_options=RLEnvironmentOptions(
            async_resets=False,
            num_respawns_per_scenario=0,
            loop_scenarios=True,
            preprocessor=chain_preprocessors(*(create_scenario_filterers() + create_scenario_preprocessors())),
            render_on_step=cfg["render_on_step"],
            render_debug_overlays=cfg["render_debug_overlays"],
            renderer_options=RENDERER_OPTIONS,
            raise_exceptions=cfg["raise_exceptions"],
            observer=FlattenedGraphObserver(data_padding_size=cfg["data_padding_size"]),
        )
    )
    return experiment_config


def configure_model(cfg: dict) -> RLModelConfig:
    reactive_planner_config = cfg["reactive_planner"]
    return RLModelConfig(
        agent_cls=ReactivePlannerAgent,  # use custom agent
        agent_kwargs=dict(
            gae_lambda=cfg["gae_lambda"],
            gamma=cfg["gamma"],
            n_epochs=cfg["n_epochs"],
            ent_coef=cfg["ent_coef"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            learning_rate=cfg["learning_rate"],
            # clip_range=cfg["clip_range"],
            # clip_range_vf=None, # not used in ReactivePlannerAgent
            policy='MultiInputPolicy',
            policy_kwargs=dict(
                ortho_init=False,
                log_std_init=-1,
                net_arch={'vf': [256, 128, 64], 'pi': [256, 128, 64]},
                activation_fn=nn.Tanh,
                features_extractor_class=VehicleGraphFeatureExtractor,
                features_extractor_kwargs=dict(
                    gnn_hidden_dim=cfg["gnn_hidden_dim"],
                    gnn_layers=cfg["gnn_layers"],
                    gnn_out_dim=cfg["gnn_out_dim"],
                    concat_ego_features=True,
                    self_loops=False,
                    aggr='max',
                    activation_fn=nn.Tanh,
                    normalization=False,
                    weight_decay=0.001
                ),
                optimizer_class=Adam,
                optimizer_kwargs=dict(
                    eps=1e-5
                )
            ),
            actor_options=ReactivePlannerActorOptions(reactive_planner_config),
        ),
    )


class InteractivePlanner:

    def __init__(self):
        # get current scenario info. from CR
        self.slow_down_due_to_intersection = None
        self.not_use_world_state = None
        self.obstacle_id = None
        self.ego_state = None
        self.world_state = None
        self.planning_cycle = None
        self.close_to_goal = None
        self.optimal = None
        self.brake = None
        self.route = None
        self.dt = None
        self.lanelet_ego = None  # the lanelet which ego car is located in
        self.lanelet_state = None  # straight-going /incoming /in-intersection
        self.lanelet_route = None
        # initialize the last action info
        self.is_new_action_needed = True
        # self.last_action = Ipaction()
        self.last_action = None
        self.last_semantic_action = None

        self.next_states_queue = []
        # goal infomation. [MCTs目标是否为goal_region, frenet中线(略)，中线距离(略)，目标位置]
        self.goal_info = None
        self.is_reach_goal_region = False
        self.initial_route = None
        self.route_time_step = 0

        # 添加路口停车相关的状态变量
        self.intersection_stop_time = 0  # 路口停车计时器
        self.is_waiting_at_intersection = False  # 是否正在路口等待
        self.intersection_wait_duration = 0.5  # 路口等待时间(秒)

    def check_state(self):
        """check if ego car is straight-going /incoming /in-intersection"""
        lanelet_id_ego = self.lanelet_ego
        ln = self.scenario.lanelet_network
        # find current lanelet
        potential_ego_lanelet_id_list = \
            self.scenario.lanelet_network.find_lanelet_by_position([self.ego_state.position])[0]
        for idx in potential_ego_lanelet_id_list:
            if idx in self.lanelet_route:
                lanelet_id_ego = idx
        self.lanelet_ego = lanelet_id_ego
        logger.info(f'current lanelet id: {self.lanelet_ego}')

        for idx_inter, intersection in enumerate(ln.intersections):
            incomings = intersection.incomings

            for idx_inc, incoming in enumerate(incomings):
                incoming_lanelets = list(incoming.incoming_lanelets)
                in_intersection_lanelets = list(incoming.successors_straight) + \
                                           list(incoming.successors_right) + list(incoming.successors_left)

                for laneletid in incoming_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 2  # incoming

                for laneletid in in_intersection_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 3  # in-intersection

        if self.lanelet_state is None:
            self.lanelet_state = 1  # straighting-going

    def check_goal_state(self, position, goal_lanelet_ids):
        is_reach_goal_lanelets = False
        ego_lanelets = self.scenario.lanelet_network.find_lanelet_by_position([position])[0]

        # goal_lanelet_ids need to change from dict to list.
        # eg. goal_lanelet_ids={0: [212], 1: [213, 214]}
        goal_lanelet_ids_list = []
        for value in goal_lanelet_ids.values():
            goal_lanelet_ids_list = goal_lanelet_ids_list + value

        for ego_lanelet in ego_lanelets:
            if ego_lanelet in goal_lanelet_ids_list:
                is_reach_goal_lanelets = True

        return is_reach_goal_lanelets

    def convert_to_state_list(self, reactive_planner_states):
        converted_states = []

        for rp_state in reactive_planner_states:
            # 创建新的State对象,只保留需要的4个参数
            new_state = CustomState(
                position=rp_state.position,
                velocity=rp_state.velocity,
                acceleration=rp_state.acceleration if hasattr(rp_state, 'acceleration') else 0,
                orientation=rp_state.orientation if hasattr(rp_state, 'orientation') else 0,
                time_step=rp_state.time_step if hasattr(rp_state, 'time_step') else 0,  # time_step 通常也需要保留
                # jerk=rp_state.jerk if hasattr(rp_state, 'jerk') else 0,
                yaw_rate=rp_state.yaw_rate if hasattr(rp_state, 'yaw_rate') else 0,
                # slip_angle=rp_state.slip_angle if hasattr(rp_state, 'slip_angle') else 0,
                steering_angle=rp_state.steering_angle if hasattr(rp_state, 'steering_angle') else 0,
            )

            converted_states.append(new_state)

        return converted_states

    def create_state_list(self, x_list, y_list, v_list, yaw_list):
        """
        Create a list of CustomState objects from given position, velocity and yaw data

        Args:
            x_list (list): List of x coordinates
            y_list (list): List of y coordinates
            v_list (list): List of velocities
            yaw_list (list): List of yaw angles

        Returns:
            list: List of CustomState objects
        """
        if not (len(x_list) == len(y_list) == len(v_list) == len(yaw_list)):
            raise ValueError("All input lists must have the same length")

        state_list = []

        for x, y, v, yaw in zip(x_list, y_list, v_list, yaw_list):
            position = np.array([float(x), float(y)])
            state = CustomState(
                position=position,
                velocity=float(v),
                yaw=float(yaw)
            )
            state_list.append(state)

        return state_list

    def visualize_trajectory(self, ego_vehicle, reactive_planner, config, current_timestep):
        if config.debug.show_plots or config.debug.save_plots:
            sampled_trajectory_bundle = None
            if config.debug.draw_traj_set:
                sampled_trajectory_bundle = copy.deepcopy(reactive_planner.stored_trajectories)
            visualize_planner_at_timestep(scenario=config.scenario,
                                          planning_problem=config.planning_problem,
                                          ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                          ref_path=reactive_planner.reference_path,
                                          # timestep=ego_vehicle.prediction.trajectory.final_state.time_step,
                                          timestep=current_timestep,
                                          config=config)

    def emergency_brake(self, current_count, factor=1):
        logger.info(f"==== perform emergency brake with factor: {factor} ====")
        self.planning_cycle = -1

        next_state = copy.deepcopy(self.ego_state)
        vehicle_params = VehicleParameterMapping.from_vehicle_type(self.vehicle_type)
        a = - vehicle_params.longitudinal.a_max / factor
        dt = self.dt
        if next_state.velocity > 0:
            v = copy.deepcopy(next_state.velocity)
            x, y = next_state.position
            o = next_state.orientation

            next_state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
            next_state.velocity += a * dt
            if next_state.velocity < 0:
                next_state.velocity = 0
                next_state.acceleration = v / dt
        next_state.acceleration = 0
        next_state.steering_angle = 0.0
        next_state.yaw_rate = 0.0
        next_state.time_step = current_count + 1
        # ====== end of motion planner

        # update the ego vehicle with new trajectory with only 1 state for the current step
        next_state = self.convert_to_state_list([next_state])[0]
        return next_state

    def emergency_brake_with_reactive_planner(self, planner, sumo_ego_vehicle, factor):
        """使用reactive planner执行紧急制动

        Args:
            planner: ReactivePlanner实例
            sumo_ego_vehicle: 自车实例
            dt: 时间步长

        Returns:
            None: 直接修改planner的期望速度
        """
        logger.info("Trying to use emergency brake with reactive planner")

        # 获取车辆参数
        vehicle_params = VehicleParameterMapping.from_vehicle_type(self.vehicle_type)
        a_max = vehicle_params.longitudinal.a_max

        # 计算减速量
        delta_v = self.dt * planner.config.planning.time_steps_computation * a_max / factor

        # 计算期望速度
        current_velocity = sumo_ego_vehicle.current_state.velocity
        desired_velocity = current_velocity - delta_v if current_velocity > delta_v else 0

        # stopping = True if current_velocity < 10 else False
        stopping = True

        # 设置planner的期望速度
        planner.set_desired_velocity(
            desired_velocity=desired_velocity,
            current_speed=planner.x_0.velocity,
            stopping=stopping
        )

        self.cost_function.w_a = 0

    def check_forward_curvature(self, distance: float = 20.0, curvature_threshold: float = 0.05) -> bool:
        """
        基于route的path_curvature检查前方指定距离是否有明显的弯道

        Args:
            distance: 前瞻距离(米)
            curvature_threshold: 曲率阈值，大于此值认为是弯道

        Returns:
            bool: True表示前方有明显弯道，False表示前方无明显弯道
        """
        try:
            if self.route is None:
                logger.warning("Route not available")
                return True  # 安全起见返回True

            # 获取必要数据
            curvatures = self.route.path_curvature
            ref_path = self.route.reference_path
            point_distances = self.route.interpoint_distances

            # 找到当前位置最近的参考点
            distances = np.linalg.norm(ref_path - self.ego_state.position, axis=1)
            current_idx = np.argmin(distances)

            # 从当前点开始累积距离
            accumulated_dist = 0.0
            look_ahead_indices = []
            idx = current_idx

            while idx < len(ref_path) - 1 and accumulated_dist < distance:
                look_ahead_indices.append(idx)
                accumulated_dist += point_distances[idx + 1]  # 使用预计算的点间距离
                idx += 1

            if look_ahead_indices:
                # 获取这段路径上的曲率
                forward_curvatures = np.abs(curvatures[look_ahead_indices])
                max_curvature = np.max(forward_curvatures)

                logger.info(f"前方{accumulated_dist:.1f}m最大曲率: {max_curvature:.3f}")

                # 检查是否有大曲率
                has_curve = max_curvature > curvature_threshold
                if has_curve:
                    logger.info("发现前方有弯道")
                else:
                    logger.info("前方道路平直")
                return has_curve

            else:
                logger.warning(f"前方{distance}m内未找到参考点")
                return True  # 安全起见返回True

        except Exception as e:
            logger.warning(f"前方道路曲率检查失败: {e}")
            return True  # 安全起见返回True

    def find_nearest_route_point_to_goal(self, goal_center) -> float:
        """
        找到参考路径上距离目标中心最近的点的弧长(s)坐标

        Args:
            goal_center: 目标区域中心点坐标, np.array([x, y])

        Returns:
            float: 最近点的弧长坐标(s)
        """
        # 参考路径上的所有点
        ref_points = self.route.reference_path

        # 计算每个点到目标中心的距离
        distances = np.linalg.norm(ref_points - goal_center, axis=1)

        # 找到最近点的索引
        nearest_idx = np.argmin(distances)

        # 返回该点的弧长坐标
        return self.route.path_length_per_point[nearest_idx]

    def plan_goal_stopping(self, planner, goal_center, distance_to_goal, ego_velocity, ttc):
        """
        改进的目标点停车规划

        Args:
            planner: ReactivePlanner实例
            goal_center: 目标区域中心点
            distance_to_goal: 到目标的距离
            ego_velocity: 当前速度
            ttc: 到达目标的预估时间
        """
        planner.set_cost_function(SafeBrakeCostFunction())
        try:
            # 找到参考路径上最近的点的弧长
            goal_s = self.find_nearest_route_point_to_goal(goal_center)

            # 根据距离和速度确定停车规划的范围
            if distance_to_goal < 30:  # 距离目标30m内开始考虑停车
                # 计算当前位置的弧长坐标
                current_s = planner._compute_initial_states(self.ego_state)[0][0]
                remaining_distance = goal_s - current_s

                # 根据剩余距离动态调整停车规划参数
                if abs(remaining_distance) < 0.3:  # 非常接近目标点
                    delta_s_min = -0.5  # 允许轻微倒车调整
                    delta_s_max = 0.5
                    # 如果速度接近0且位置合适，允许完全停止
                    if ego_velocity < 0.05 and abs(remaining_distance) < 0.1:
                        planner.set_desired_velocity(
                            desired_velocity=0.0,
                            current_speed=ego_velocity,
                            stopping=True
                        )
                        return True
                elif abs(remaining_distance) < 2.0:  # 比较接近目标点
                    delta_s_min = -1.0
                    delta_s_max = max(1.0, remaining_distance * 1.2)
                    # 降低但不要完全停止
                    planner._desired_speed = min(0.3, ego_velocity)
                else:  # 还有一定距离
                    # 根据当前速度预留足够的停车距离
                    stopping_distance = ego_velocity * min(2.0, ttc)  # 使用ttc但限制最大值
                    delta_s_min = -stopping_distance * 0.1  # 允许少量倒车
                    delta_s_max = stopping_distance * 1.2  # 预留20%的缓冲距离
                    # 逐步降低期望速度
                    target_speed = max(0.5, min(ego_velocity,
                                                abs(remaining_distance) / 3.0))  # 根据剩余距离计算合适的速度
                    planner.set_desired_velocity(current_speed=ego_velocity, desired_velocity=target_speed)

                logger.info(f"Planning stop at s={goal_s:.2f}m with range [{delta_s_min:.2f}, {delta_s_max:.2f}], "
                            f"remaining distance={remaining_distance:.2f}m, target speed={planner._desired_speed:.2f}m/s")

                # 调整cost function权重，使其更注重位置精度
                if hasattr(planner.cost_function, "w_position"):
                    if abs(remaining_distance) < 5.0:
                        planner.cost_function.w_position *= 100.0  # 增加位置权重
                        planner.cost_function.w_lateral_position *= 100.0  # 增加横向位置权重

                # 使用ReactivePlanner的停车规划功能
                planner.set_desired_lon_position(
                    lon_position=goal_s,
                    delta_s_min=delta_s_min,
                    delta_s_max=delta_s_max
                )
                return True

        except Exception as e:
            logger.warning(f"停车规划失败: {str(e)}")
            return False

        except Exception as e:
            logger.warning(f"停车规划失败: {str(e)}")
            return False

    def process_goal_position(self, planning_problem, sumo_ego_vehicle, planner, current_count):
        """
        处理目标位置相关的停车规划逻辑

        Args:
            planning_problem: 规划问题
            sumo_ego_vehicle: 自车状态
            planner: ReactivePlanner实例
            current_count: 当前时间步

        Returns:
            next_state: 如果需要紧急制动返回next_state，否则返回None
        """
        try:
            goal_center = (planning_problem.goal.state_list[0].position.center if hasattr(
                planning_problem.goal.state_list[0].position, 'center')
                           else planning_problem.goal.state_list[0].position.shapes[0].center if hasattr(
                planning_problem.goal.state_list[0].position, 'shapes')
            else None)

            goal_area = (planning_problem.goal.state_list[0].position.shapely_object.area if hasattr(
                planning_problem.goal.state_list[0].position, 'center')
                         else planning_problem.goal.state_list[0].position.shapes[0].shapely_object.area if hasattr(
                planning_problem.goal.state_list[0].position, 'shapes')
            else None)

            if goal_center is not None and goal_area < 200:
                distance_to_goal = np.linalg.norm(goal_center - sumo_ego_vehicle.current_state.position)
                ego_velocity = sumo_ego_vehicle.current_state.velocity

                # 获取目标速度(如果有的话)
                goal_velocity = (planning_problem.goal.state_list[0].velocity.end
                                 if 'velocity' in planning_problem.goal.state_list[0].__dict__.keys()
                                 else 0)
                # 计算速度差
                velocity_diff = ego_velocity - goal_velocity
                # 计算时间到达目标
                ttc = distance_to_goal / (velocity_diff + 1e-5)  # 防止除零

                if distance_to_goal < 5:
                    self.is_reach_goal_region = True

                # 判断是否需要刹车
                if distance_to_goal < 10 or self.is_reach_goal_region:
                    self.close_to_goal = True
                    logger.info(
                        f'=== 接近目标点 (距离: {distance_to_goal:.2f}m)), 规划使用lon_position停车 ===')
                    # 尝试使用平滑停车规划
                    if self.plan_goal_stopping(planner, goal_center, distance_to_goal, ego_velocity, ttc):
                        self.brake = True
                    else:
                        # 如果平滑停车规划失败，回退到原来的紧急制动逻辑
                        logger.info("平滑停车规划失败，使用常规制动")
                        self.brake = True

                elif distance_to_goal < ego_velocity * 3 and (ttc < 3 or velocity_diff > 15):
                    logger.info(
                        f'=== 接近目标点 (距离: {distance_to_goal:.2f}m) 速度过高 ({ego_velocity:.2f}m/s), 执行制动 ===')
                    self.brake = True

            return None

        except Exception as e:
            logger.info(f"处理目标位置相关停车规划失败: {str(e)}")
            return None

    def get_speed_limitation(self,
                             scenario,
                             current_lanelet,
                             current_speed,
                             current_count,
                             planning_problem) -> tuple[float, float]:
        """
        获取当前车道的限速值和建议速度

        Args:
            scenario: 场景对象
            sumo_ego_vehicle: 自车状态

        Returns:
            tuple: (speed_limitation, desired_velocity)
            - speed_limitation: 限速值，如果没有限速则返回None
            - desired_velocity: 建议速度，考虑了限速和其他因素
        """
        speed_limitation = None

        try:
            lanelet_network = scenario.lanelet_network

            obstacle_speed = []
            for obstacle in scenario.dynamic_obstacles:
                # 判断obstacle和ego距离是否小于50m
                if np.linalg.norm(obstacle.initial_state.position - self.ego_state.position) < 25:
                    obstacle_speed.append(obstacle.initial_state.velocity)
            speed_limitation_not_designated = max(min(max(obstacle_speed) * 1.2, 35), 15) if obstacle_speed else 30

            # 如果时间不够，可以稍微增加限速
            goal_time_step = self.num_of_steps
            if 'current_count' in planning_problem.goal.state_list[0].__dict__.keys():
                if hasattr(planning_problem.goal.state_list[0].current_count, 'start'):
                    goal_time_step = planning_problem.goal.state_list[0].current_count.start
            if 0 < goal_time_step - current_count < 50:
                speed_limitation_not_designated = max(speed_limitation_not_designated * 1.2, 20)

            # 检查是否有交通标志
            if current_lanelet.traffic_signs:
                speed_limitations = []

                # 遍历所有交通标志
                for traffic_sign_id in current_lanelet.traffic_signs:
                    traffic_sign = lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
                    traffic_sign_element = traffic_sign.traffic_sign_elements[0]

                    # 检查是否是限速标志
                    if traffic_sign_element.traffic_sign_element_id.name == 'MAX_SPEED':
                        speed_limitations.append(float(traffic_sign_element.additional_values[0]))

                # 如果有限速标志
                if speed_limitations:
                    # 限制最高速度为40m/s
                    speed_limitation = min(40, min(speed_limitations))
                    logger.info(f"speed limitation: {speed_limitation}")

                    # 如果不接近目标，可以稍微超速
                    if not self.close_to_goal:
                        desired_velocity = speed_limitation
                    else:
                        desired_velocity = speed_limitation
                else:
                    logger.info(f"no speed limitation, set speed limitation to {max(speed_limitation_not_designated, current_speed)}")
                    desired_velocity = max(speed_limitation_not_designated, current_speed)
            else:
                logger.info(f"no speed limitation，set speed limitation to {max(speed_limitation_not_designated, current_speed)}")
                desired_velocity = max(speed_limitation_not_designated, current_speed)

        except Exception as e:
            logger.warning(f"*** get speed limitation failed ***")
            logger.warning(f"failed reason: {e}")
            # 发生异常时使用默认值
            speed_limitation = 20
            desired_velocity = 20

        return speed_limitation, desired_velocity

    def check_and_replan_route(self, scenario, planning_problem, planner, current_position) -> bool:
        """
        检查当前位置并在必要时重新规划路径

        Args:
            scenario: 场景对象
            planning_problem: 规划问题
            planner: ReactivePlanner实例
            current_position: 当前位置 [x, y]

        Returns:
            bool: 是否进行了重新规划
        """
        try:
            lanelet_network = scenario.lanelet_network
            lanelet_id_ego = None

            # 找到当前位置的所有可能车道
            potential_ego_lanelet_id_list = lanelet_network.find_lanelet_by_position([current_position])[0]

            # 优先选择规划路径上的车道
            for idx in potential_ego_lanelet_id_list:
                if idx in self.lanelet_route:
                    lanelet_id_ego = idx
                    break

            # 如果接近路口或在交叉口内，取消Route重规划
            if self.lanelet_state in [2, 3]:
                logger.info("Close to intersection or in intersection, cancel route replanning")
                return False

            # 检查是否需要重新规划
            route_replanned = False
            if lanelet_id_ego is None:
                # 使用当前位置的第一个车道
                lanelet_id_ego = potential_ego_lanelet_id_list[0]

                # 重新规划路径
                route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
                self.route = route_planner.plan_routes().retrieve_first_route()
                self.lanelet_route = self.route.lanelet_ids

                # 更新ReactivePlanner的参考路径
                planner.set_reference_path(self.route.reference_path)

                route_replanned = True
                logger.info("!!! Route replanned due to vehicle not on planned route !!!")

            # 更新当前车道ID
            self.lanelet_ego = lanelet_id_ego
            logger.info(f'current lanelet id: {self.lanelet_ego}')

            return route_replanned

        except Exception as e:
            logger.warning(f"Route replanning failed: {str(e)}")
            return False

    def stop_before_intersection(self, scenario, planner, current_position, sumo_ego_vehicle, low_speed_threshold= 1) -> bool:
        """在接近交叉口时控制车辆停车

        Args:
            scenario: 场景对象，包含道路网络信息
            planning_problem: 规划问题对象，包含目标状态信息
            planner: 规划器对象
            current_position: 当前位置
            sumo_ego_vehicle: ego车辆对象

        Returns:
            bool: 是否成功处理交叉口停车逻辑
        """
        try:
            if sumo_ego_vehicle.current_state.velocity < low_speed_threshold:
                logger.info("车速过低，不执行交叉口停车逻辑，启动车辆")
                self.brake = False
                return False

            # 1. 基础检查
            lanelet_network = scenario.lanelet_network
            if not lanelet_network.intersections:
                return False

            # 2. 获取当前位置可能的车道ID
            potential_ego_lanelet_ids = lanelet_network.find_lanelet_by_position([current_position])[0]
            if not potential_ego_lanelet_ids:
                return False

            # 3. 查找交叉口终点
            lanelet_end_point = None
            for intersection in lanelet_network.intersections:
                for incoming in intersection.incomings:
                    # 检查当前车道是否在交叉口入口车道中
                    matching_lanelet = next(
                        (lanelet_id for lanelet_id in incoming.incoming_lanelets
                         if lanelet_id in potential_ego_lanelet_ids),
                        None
                    )
                    if matching_lanelet:
                        lanelet = lanelet_network.find_lanelet_by_id(matching_lanelet)
                        lanelet_end_point = lanelet.center_vertices[-1]
                        logger.info(f"即将进入交叉口，车道终点: {lanelet_end_point}")
                        break
                if lanelet_end_point is not None:
                    break

            if lanelet_end_point is None:
                return False

            # 4. 计算关键参数
            ego_state = sumo_ego_vehicle.current_state
            distance_to_goal = np.linalg.norm(lanelet_end_point - ego_state.position)
            ego_velocity = ego_state.velocity

            # 获取目标速度
            goal_velocity = 0

            # 计算到达时间
            velocity_diff = ego_velocity - goal_velocity
            ttc = distance_to_goal / (velocity_diff + 1e-5)  # 防止除零

            # 5. 更新状态和执行停车策略
            if distance_to_goal < 10:
                # 非常接近交叉口，执行平滑停车
                logger.info(f'=== 接近路口 (距离: {distance_to_goal:.2f}m), 规划使用lon_position停车 ===')

                if self.plan_goal_stopping(planner, lanelet_end_point, distance_to_goal, ego_velocity, ttc):
                    self.brake = True
                    self.slow_down_due_to_intersection = True
                    return True
                else:
                    logger.info("平滑停车规划失败，使用常规制动")
                    self.brake = True
                    self.slow_down_due_to_intersection = True
                    return False

            elif distance_to_goal < ego_velocity * 3 and (ttc < 3 or velocity_diff > 15):
                # 距离较远但速度过高，需要提前制动
                logger.info(
                    f'=== 接近路口 (距离: {distance_to_goal:.2f}m) 速度过高 ({ego_velocity:.2f}m/s), 执行制动 ===')
                self.brake = True
                self.slow_down_due_to_intersection = True
                return False

        except Exception as e:
            logger.warning(f"停车控制失败: {str(e)}")
            return False

    def compare_states(self, state1, state2, threshold=0.1):
        """
        比较两个状态并报告差异

        Args:
            state1, state2: 要比较的状态
            threshold: 差异阈值
        """
        diff_items = []  # 存储所有超过阈值的项

        # 比较position
        diff_x = abs(state1.position[0] - state2.position[0])
        diff_y = abs(state1.position[1] - state2.position[1])
        if diff_x > threshold:
            diff_items.append(f"position_x: {diff_x:.6f}")
        if diff_y > threshold:
            diff_items.append(f"position_y: {diff_y:.6f}")

        # 比较其他属性
        for attr in ["steering_angle", "velocity", "orientation", "acceleration", "yaw_rate"]:
            diff = abs(getattr(state1, attr) - getattr(state2, attr))
            if diff > threshold:
                diff_items.append(f"{attr}: {diff:.6f}")

        # 如果有差异，输出警告
        if diff_items:
            logger.warning(f"********** Initial state differs in: {', '.join(diff_items)} **********")

    def initialize(self, folder_scenarios, name_scenario):
        self.vehicle_type = VehicleType.FORD_ESCORT
        self.vehicle_model = VehicleModel.KS
        self.cost_function = CostFunction.TR1
        self.vehicle = VehicleDynamics.KS(self.vehicle_type)

        interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

        conf = load_sumo_configuration(interactive_scenario_path)
        scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
        try:
            self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_file).open()

            scenario_wrapper = ScenarioWrapper()
            scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
            scenario_wrapper.initial_scenario = self.scenario

            self.num_of_steps = conf.simulation_steps
            sumo_sim = SumoSimulation()

            # initialize simulation
            sumo_sim.initialize(conf, scenario_wrapper, self.planning_problem_set)

            self.t_record = 0

        except:
            logger.info(f"*** failed to initialize the scenariom, try it again ***")
            self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_file).open()

            scenario_wrapper = ScenarioWrapper()
            scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
            scenario_wrapper.initial_scenario = self.scenario

            self.num_of_steps = conf.simulation_steps
            sumo_sim = SumoSimulation()

            # initialize simulation
            sumo_sim.initialize(conf, scenario_wrapper, self.planning_problem_set)

            self.t_record = 0

        return sumo_sim

    def process(self, sumo_sim, cfg_obj):

        # generate ego vehicle
        sumo_ego_vehicles = sumo_sim.ego_vehicles

        # configure experiment and model
        model_config = configure_model(Config(cfg_obj.model))

        # noinspection PyTypeChecker
        reactive_planner_config = _dict_to_params(
            OmegaConf.create(model_config.agent_kwargs['actor_options'].reactive_planner_options.as_dict()),
            ReactivePlannerConfiguration
        )

        self.dt = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step).dt

        planning_problem_set = copy.deepcopy(self.planning_problem_set)
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        lanelets_of_goal_position = copy.deepcopy(planning_problem.goal.lanelets_of_goal_position)

        # initial positions do not match
        sumo_ego_vehicle = list(sumo_ego_vehicles.values())[0]
        sumo_ego_vehicle.current_state.position = copy.deepcopy(planning_problem.initial_state.position)
        sumo_ego_vehicle.current_state.orientation = copy.deepcopy(planning_problem.initial_state.orientation)
        sumo_ego_vehicle.current_state.velocity = copy.deepcopy(planning_problem.initial_state.velocity)
        sumo_ego_vehicle.current_state.acceleration = copy.deepcopy(planning_problem.initial_state.acceleration)
        sumo_ego_vehicle.current_state.steering_angle = 0.0
        sumo_ego_vehicle.current_state.yaw_rate = 0.0

        scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
        reactive_planner_config.update(scenario=scenario, planning_problem=planning_problem)

        route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
        self.route = route_planner.plan_routes().retrieve_first_route()
        self.lanelet_route = self.route.lanelet_ids

        planner = ReactivePlanner(reactive_planner_config)
        planner.set_reference_path(self.route.reference_path)
        planner.record_state_and_input(planner.x_0)

        self.planning_cycle = 0

        logger.info("=" * 50 + " Start Simulation " + "=" * 50)
        for current_count in range(self.num_of_steps):
        # for current_count in range(0, 300):

            logger.info(f"process: {current_count}/{self.num_of_steps}")

            current_scenario = sumo_sim.commonroad_scenario_at_time_step(current_count)
            for obstacle in current_scenario.dynamic_obstacles:
                obstacle.initial_state.time_step = current_count
                predicted_states = predict_constant_velocity(obstacle.initial_state, self.dt,
                                                             planner.config.planning.time_steps_computation)
                trajectory = Trajectory(current_count, [obstacle.initial_state] + predicted_states)
                obstacle.prediction = TrajectoryPrediction(trajectory, shape=obstacle.obstacle_shape)
                if current_count == 0:
                    obstacle.obstacle_shape.length += 0.5
                    obstacle.obstacle_shape.width += 0.1
            planner.set_collision_checker(scenario=current_scenario)
            planner.config.scenario = current_scenario
            planner.loose_constraints = False

            self.ego_state = sumo_ego_vehicle.current_state

            try:
                # generate a CR planner
                next_state = self.planning(copy.deepcopy(current_scenario),
                                           planning_problem_set,
                                           sumo_ego_vehicle,
                                           current_count,
                                           reactive_planner_config,
                                           lanelets_of_goal_position,
                                           planner)
            except Exception as e:
                # 如果整个规划过程失败，执行紧急制动
                logger.info(f"*** failed to plan the scenario ***")
                logger.info("Error: " + str(e))
                next_state = self.emergency_brake(current_count)
                next_state_shifted = next_state.translate_rotate(
                    np.array([-self.vehicle.parameters.b * np.cos(next_state.orientation),
                              -self.vehicle.parameters.b * np.sin(next_state.orientation)]),
                    0.0)
                planner.record_state_and_input(next_state_shifted)
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              collision_checker=planner.collision_checker,
                              coordinate_system=planner.coordinate_system)

            self.planning_cycle += 1

            logger.info(
                f'next ego position: {next_state.position}, next ego steering angle: {next_state.steering_angle}, next ego velocity: {next_state.velocity}')
            logger.info(
                f'next ego orientation: {next_state.orientation}, next ego acceleration: {next_state.acceleration}, next yaw rate: {next_state.yaw_rate}')

            feasible, reconstructed_input = state_transition_feasibility(self.ego_state, next_state,
                                                                         self.vehicle, self.dt)
            if not feasible:
                logger.info("state transition infeasible, trying reconstruct")
                vehicle = self.vehicle
                next_state_reconstructed = vehicle.simulate_next_state(self.ego_state, reconstructed_input, self.dt)
                logger.info(f"reconstructed state: {next_state_reconstructed}")
                # feasible, _ = state_transition_feasibility(self.ego_state, next_state_reconstructed, vehicle, self.dt)
                # logger.info(f"feasible: {feasible}")

                next_state.position = next_state_reconstructed.position
                next_state.velocity = next_state_reconstructed.velocity
                next_state.orientation = next_state_reconstructed.orientation
                next_state.steering_angle = next_state_reconstructed.steering_angle

                next_state_shifted = next_state.translate_rotate(
                    np.array([-self.vehicle.parameters.b * np.cos(next_state.orientation),
                              -self.vehicle.parameters.b * np.sin(next_state.orientation)]),
                    0.0)
                planner.record_input_list.pop()
                planner.record_state_list.pop()
                planner.record_state_and_input(next_state_shifted)
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              collision_checker=planner.collision_checker,
                              coordinate_system=planner.coordinate_system)

                self.planning_cycle = 0

            # ====== paste in simulations
            # ====== end of motion planner
            next_state.time_step = 1
            # next_state.steering_angle = 0.0
            trajectory_ego = [next_state]
            sumo_ego_vehicle.set_planned_trajectory(trajectory_ego)

            sumo_sim.simulate_step()
            logger.info("=" * 120)

        # retrieve the simulated scenario in CR format
        simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

        # stop the simulation
        sumo_sim.stop()

        # match pp_id
        ego_vehicles = {list(self.planning_problem_set.planning_problem_dict.keys())[0]:
                            ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

        for pp_id, planning_problem in self.planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            simulated_scenario.add_objects(obstacle_ego)

        return simulated_scenario, ego_vehicles

    def planning(
            self,
            scenario,
            planning_problem_set,
            sumo_ego_vehicle,
            current_count,
            reactive_planner_config,
            lanelets_of_goal_position,
            planner
    ):
        logger.info(
            f"ego position: {sumo_ego_vehicle.current_state.position}, ego steering angle: {sumo_ego_vehicle.current_state.steering_angle}, ego velocity: {sumo_ego_vehicle.current_state.velocity}")
        logger.info(
            f"ego orientation: {sumo_ego_vehicle.current_state.orientation}, ego acceleration: {sumo_ego_vehicle.current_state.acceleration}, yaw rate: {sumo_ego_vehicle.current_state.yaw_rate}")

        # 判断自车车道状态
        self.check_state()
        logger.info(f"lanelet state: {self.lanelet_state}")

        # 判断是否到达目标
        try:
            if not self.is_reach_goal_region:
                self.is_reach_goal_region = self.check_goal_state(sumo_ego_vehicle.current_state.position,
                                                                  lanelets_of_goal_position)
        except:
            pass

        # 接近目标刹车
        self.brake = False
        if self.is_reach_goal_region:
            logger.info('*** 到达目标 ***')
            self.planning_cycle = 0

            # 如果到达目标且速度为0，直接返回当前状态
            if sumo_ego_vehicle.current_state.velocity == 0:
                next_state = copy.deepcopy(self.ego_state)
                next_state.time_step = 1
                return next_state

            # 如果到达目标且速度不为0，执行停车
            self.brake = True

        # 检查并重新规划路径
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        self.check_and_replan_route(
            scenario,
            planning_problem,
            planner,
            sumo_ego_vehicle.current_state.position
        )
        current_lanelet = scenario.lanelet_network.find_lanelet_by_id(self.lanelet_ego)

        # check if planning cycle or not
        plan_new_trajectory = self.planning_cycle % reactive_planner_config.planning.replanning_frequency == 0

        if plan_new_trajectory:

            # 接近目标减速
            if 'position' not in planning_problem.goal.state_list[0].__dict__.keys():
                # 如果没有位置目标，直接跳过位置相关判断
                pass
            elif 'velocity' not in planning_problem.goal.state_list[0].__dict__.keys():
                if 'time_step' in planning_problem.goal.state_list[0].__dict__.keys():
                    if planning_problem.goal.state_list[0].time_step.start > 50:
                        # 处理目标位置相关的停车规划
                        next_state = self.process_goal_position(planning_problem, sumo_ego_vehicle, planner,
                                                                current_count)
                        if next_state is not None:
                            return next_state
            else:
                # 处理目标位置相关的停车规划
                next_state = self.process_goal_position(planning_problem, sumo_ego_vehicle, planner, current_count)
                if next_state is not None:
                    return next_state

            # 处理交叉口停车
            close_to_intersection = False
            self.slow_down_due_to_intersection = False
            try:
                close_to_intersection = self.stop_before_intersection(scenario, planner,
                                                                      sumo_ego_vehicle.current_state.position,
                                                                      sumo_ego_vehicle,
                                                                      low_speed_threshold=3)
                if self.slow_down_due_to_intersection:
                    goal_time_step = self.num_of_steps
                    if 'current_count' in planning_problem.goal.state_list[0].__dict__.keys():
                        if hasattr(planning_problem.goal.state_list[0].current_count, 'start'):
                            goal_time_step = planning_problem.goal.state_list[0].current_count.start
                    if 0 < goal_time_step - current_count < 50:
                        close_to_intersection = False
                        self.brake = False
                        logger.info(f"=== 时间不够，不执行交叉口停车逻辑 ===")
            except Exception as e:
                logger.warning(f"*** failed to detect stopping before intersection ***")
            logger.info(f"=== close_to_intersection: {close_to_intersection} ===")

            # 获取限速
            speed_limitation, desired_velocity = self.get_speed_limitation(scenario, current_lanelet,
                                                                           planner.x_0.velocity,
                                                                           current_count,
                                                                           planning_problem)

            # 设置cost function
            if not self.is_reach_goal_region and not self.brake and not self.close_to_goal:

                # 获取wrold state用于robustness计算
                ego_dynamic_obstacle = None
                try:
                    if current_count > 0 and not self.not_use_world_state:
                        ego_dynamic_obstacle = sumo_ego_vehicle.get_dynamic_obstacle()
                        self.obstacle_id = ego_dynamic_obstacle.obstacle_id
                        self.world_state = World.create_from_scenario(scenario)
                        self.world_state.add_controlled_vehicle(
                            self.scenario, copy.deepcopy(ego_dynamic_obstacle)
                        )
                        logger.info("获取world state成功")
                except Exception as e:
                    logger.info(f"*** failed to create world state ***")
                    logger.info("Error: " + str(e))
                    self.world_state = None
                    self.not_use_world_state = True

                planner.set_cost_function(BaselineCostFunction(desired_speed=planner.desired_speed,
                                                               desired_d=0.0,
                                                               desired_s=planner.desired_lon_position,
                                                               world_state=self.world_state,
                                                               obstacle_id=self.obstacle_id,
                                                               vehicle=self.vehicle,
                                                               ego_dynamic_obstacle=copy.deepcopy(ego_dynamic_obstacle),
                                                               speed_limitation=speed_limitation if speed_limitation else None,
                                                               scenario=scenario,
                                                               current_time_step=current_count,
                                                               ))

                # 目标时间接近时，增加反低速权重
                goal_time_step = self.num_of_steps
                if 'current_count' in planning_problem.goal.state_list[0].__dict__.keys():
                    if hasattr(planning_problem.goal.state_list[0].current_count, 'start'):
                        goal_time_step = planning_problem.goal.state_list[0].current_count.start
                if 0 < goal_time_step - current_count < 30:
                    planner.cost_function.w_low_speed = max(5e3 / (goal_time_step - current_count), 1e2)
                planner.set_desired_curve_velocity(current_speed=planner.x_0.velocity,
                                                   desired_velocity=desired_velocity)

            elif self.close_to_goal or close_to_intersection:
                pass

            else:
                planner.set_cost_function(SafeBrakeCostFunction())
                self.emergency_brake_with_reactive_planner(planner, sumo_ego_vehicle, factor=2)

            # **************************
            # Run Planning
            # **************************

            try:
                # plan optimal trajectory
                self.optimal = planner.plan()
            except Exception as e:
                logger.warning(f"*** reactive_planner failed ***")
                logger.warning(f"failed reason: {e}")
                self.optimal = None

            # If failed, visualize the failed trajectory and perform emergency brake
            if not self.optimal:
                try:
                    x_0 = planner.x_0
                    rp_failure_ego_vehicle = planner.convert_state_list_to_commonroad_object([x_0])
                    self.visualize_trajectory(rp_failure_ego_vehicle, planner, reactive_planner_config,
                                              current_count)
                    self.visualize_trajectory(rp_failure_ego_vehicle, planner, reactive_planner_config,
                                              current_count + 10)
                except Exception as e:
                    logger.warning(f"visualize failed trajectory failed")
                    logger.warning(f"failed reason: {e}")

                # loose constraints and replan
                logger.info("=== Replan with loose constraints ===")
                try:
                    planner.loose_constraints = True
                    self.optimal = planner.plan()
                finally:
                    # planner.loose_constraints = False
                    pass

                if self.optimal is None:
                    # If too many infeasible trajectories due to collision, perform emergency brake
                    if planner.infeasible_count_collision > 10:
                        logger.warning("=== Too many infeasible trajectories due to collision ===")
                        planner.set_cost_function(SafeBrakeCostFunction())
                        self.emergency_brake_with_reactive_planner(planner, sumo_ego_vehicle, factor=2)
                        self.optimal = planner.plan()
                        if not self.optimal and planner.infeasible_count_collision < 10:
                            logger.warning("=== All infeasible trajectories are due to hard braking ===")
                            desired_velocity = planner.x_0.velocity - 4 if planner.x_0.velocity > 4 else 0
                            planner.set_desired_velocity(current_speed=planner.x_0.velocity,
                                                         desired_velocity=desired_velocity)
                            self.optimal = planner.plan()

                    elif self.brake:
                        logger.warning("=== All infeasible trajectories are due to hard braking ===")
                        planner.set_cost_function(SafeBrakeCostFunction())
                        desired_velocity = planner.x_0.velocity - 4 if planner.x_0.velocity > 4 else 0
                        planner.set_desired_velocity(current_speed=planner.x_0.velocity, desired_velocity=desired_velocity)
                        self.optimal = planner.plan()

                    else:
                        logger.warning("=== All infeasible trajectories are due to kinematic constraints ===")
                        planner.set_cost_function(SafeBrakeCostFunction())
                        planner.set_desired_velocity(current_speed=planner.x_0.velocity,
                                                     desired_velocity=planner.x_0.velocity)
                        self.optimal = planner.plan()

                    if self.optimal is None:
                        # logger.warning("Failed to plan optimal trajectory with fail-safe cost function, perform emergency brake")
                        next_state = self.emergency_brake(current_count)
                        next_state_shifted = next_state.translate_rotate(
                            np.array([-self.vehicle.parameters.b * np.cos(next_state.orientation),
                                      -self.vehicle.parameters.b * np.sin(next_state.orientation)]),
                            0.0)
                        planner.record_state_and_input(next_state_shifted)
                        planner.reset(initial_state_cart=planner.record_state_list[-1],
                                      collision_checker=planner.collision_checker,
                                      coordinate_system=planner.coordinate_system)
                        return next_state

            # logger.info(f"x_0_initial: {planner.x_0}")
            # logger.info(f"x_0_planned: {self.optimal[0].state_list[0]}")
            # self.compare_states(planner.x_0, self.optimal[0].state_list[0])

            # Normal process
            # record state and input
            planner.record_state_and_input(self.optimal[0].state_list[1])

            rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(self.optimal[0].state_list)
            next_states = rp_ego_vehicle.prediction.trajectory.state_list[1:]
            next_state = next_states[0]

            # reset planner state for re-planning
            planner.reset(initial_state_cart=planner.record_state_list[-1],
                          initial_state_curv=(self.optimal[2][1], self.optimal[3][1]),
                          collision_checker=planner.collision_checker,
                          coordinate_system=planner.coordinate_system)

        else:
            # simulate scenario one step forward with planned trajectory
            # continue on optimal trajectory
            temp = self.planning_cycle % reactive_planner_config.planning.replanning_frequency

            # record state and input
            planner.record_state_and_input(self.optimal[0].state_list[1 + temp])

            rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(self.optimal[0].state_list)
            next_states = rp_ego_vehicle.prediction.trajectory.state_list[1 + temp:]
            next_state = next_states[0]

            # reset planner state for re-planning
            planner.reset(initial_state_cart=planner.record_state_list[-1],
                          initial_state_curv=(self.optimal[2][1 + temp], self.optimal[3][1 + temp]),
                          collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

        try:
            if not self.is_reach_goal_region:
                self.is_reach_goal_region = planner.goal_reached()
        except Exception as e:
            logger.warning(f"*** goal_reached failed ***")
            logger.warning(f"failed reason: {e}")

        return next_state


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig):
    cfg_obj = OmegaConf.to_object(cfg)

    # folder_scenarios = os.path.abspath('/home/yanliang/commonroad-scenarios/scenarios/interactive/SUMO/')
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-interactive-scenarios/scenarios/tutorial")
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-scenarios/scenarios/interactive/hand-crafted")
    folder_scenarios = os.path.abspath("/home/yanliang/scenarios_phase")

    # name_scenario = "DEU_Frankfurt-95_6_I-1" # OK
    # name_scenario = "CHN_Sha-6_5_I-1-1" # OK
    # name_scenario = "DEU_Frankfurt-152_8_I-1" # 无法到达终点
    # name_scenario = "ESP_Mad-2_1_I-1-1" # OK

    # name_scenario = "DEU_Cologne-63_5_I-1" # OK
    # name_scenario = "DEU_Aachen-2_1_I-1"
    # name_scenario = "DEU_Aachen-3_1_I-1" # OK

    # name_scenario = "ZAM_Tjunction-1_270_I-1-1" # OK
    # name_scenario = "ZAM_Tjunction-1_350_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_517_I-1-1" # 很难解
    # name_scenario = "DEU_Ffb-2_2_I-1-1"
    # name_scenario = "DEU_A9-2_1_I-1-1" # OK
    # name_scenario = "ZAM_Zip-1_7_I-1-1" # OK
    # name_scenario = "ZAM_Tjunction-1_42_I-1-1"
    # name_scenario = "DEU_Muc-4_2_I-1-1" # OK
    # name_scenario = "ZAM_Tjunction-1_32_I-1-1"
    # name_scenario = "ZAM_Zip-1_69_I-1-1" # OK
    # name_scenario = "ZAM_ACC-1_3_I-1-1" # OK
    # name_scenario = "DEU_A9-1_2_I-1-1" # OK
    # name_scenario = "ZAM_Tjunction-1_258_I-1-1"  # 解不了
    # name_scenario = "ZAM_Tjunction-1_110_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_75_I-1-1"  # 修改了intersection信息**
    # name_scenario = "DEU_Gar-1_2_I-1-1" # OK
    # name_scenario = "ZAM_Merge-1_1_I-1-1" # 刹不住车的

    # name_scenario = "DEU_Aachen-3_1_I-1" # OK
    # name_scenario = "DEU_Aachen-3_2_I-1" # OK
    # name_scenario = "DEU_Aachen-3_3_I-1" # OK
    # name_scenario = "DEU_Aachen-3_7_I-1"  # OK
    # name_scenario = "DEU_Aachen-3_12_I-1"  # OK
    name_scenario = "DEU_Aachen-3_13_I-1"  # OK
    # name_scenario = "DEU_Aachen-29_9_I-1" # OK
    # name_scenario = "DEU_Aachen-29_12_I-1" # 速度太慢被后车追尾
    # name_scenario = "DEU_Aachen-29_13_I-1" # OK
    # name_scenario = "DEU_Aachen-29_14_I-1" # OK
    # name_scenario = "DEU_Aachen-29_15_I-1" # OK
    # name_scenario = "DEU_Aachen-33_3_I-1" # 速度太慢被后车追尾
    # name_scenario = "DEU_Aachen-33_4_I-1" # 出生位置有车，无法解
    # name_scenario = "DEU_Aachen-33_5_I-1" # OK
    # name_scenario = "DEU_Aachen-33_6_I-1" # OK
    # name_scenario = "DEU_Cologne-27_15_I-1" # OK
    # name_scenario = "DEU_Cologne-21_1_I-1" # OK
    # name_scenario = "DEU_Cologne-21_2_I-1" # OK
    # name_scenario = "DEU_Cologne-32_2_I-1" # 初始速度太高，无法解
    # name_scenario = "DEU_Cologne-32_3_I-1" # OK
    # name_scenario = "DEU_Cologne-32_4_I-1" # OK 路口直行，减速通过
    # name_scenario = "DEU_Cologne-32_5_I-1" # OK
    # name_scenario = "DEU_Cologne-32_13_I-1" # OK 初始速度太高，但是通过动态放松限制解开了
    # name_scenario = "DEU_Cologne-40_6_I-1" # OK
    # name_scenario = "DEU_Cologne-61_24_I-1" # 出生位置有车，无法解
    # name_scenario = "DEU_Cologne-61_25_I-1" # OK
    # name_scenario = "DEU_Cologne-61_26_I-1" # OK
    # name_scenario = "DEU_Cologne-63_8_I-1" # OK
    # name_scenario = "DEU_Dresden-3_14_I-1" # OK
    # name_scenario = "DEU_Dresden-3_29_I-1" # OK
    # name_scenario = "DEU_Dresden-18_4_I-1" # OK
    # name_scenario = "DEU_Dresden-18_5_I-1" # OK
    # name_scenario = "DEU_Dresden-18_6_I-1" # OK
    # name_scenario = "DEU_Dresden-18_29_I-1" # OK
    # name_scenario = "DEU_Dresden-18_30_I-1" # OK 要到达终点时路径诡异，增加约束后正常
    # name_scenario = "DEU_A99-1_1_I-1-1" # OK

    main_planner = InteractivePlanner()
    sumo_sim = main_planner.initialize(folder_scenarios, name_scenario)
    simulated_scenario, ego_vehicles = main_planner.process(sumo_sim, cfg_obj)

    # path for outputting results
    output_path = '/home/yanliang/cr-competition-outputs'

    # video
    output_folder_path = os.path.join(output_path, 'videos/')
    # solution
    path_solutions = os.path.join(output_path, 'solutions/')

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    trajectory._state_list = [ego_vehicle.initial_state] + trajectory.state_list

    # create mp4 animation
    create_video(simulated_scenario,
                 output_folder_path,
                 main_planner.planning_problem_set,
                 ego_vehicles,
                 True,
                 "_planner")

    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt,
                                                                                e=np.array([5e-2, 5e-2, 5e-2])
                                                                                )
    logger.info('Feasible? {}'.format(feasible))
    if not feasible:
        logger.info(f'infeasible point: {len(reconstructed_inputs.state_list)}')

    # saves trajectory to solution file
    save_solution(simulated_scenario, main_planner.planning_problem_set, ego_vehicles,
                  main_planner.vehicle_type,
                  main_planner.vehicle_model,
                  main_planner.cost_function,
                  path_solutions, overwrite=True)

    solution = CommonRoadSolutionReader.open(os.path.join(path_solutions,
                                                          f"solution_KS1:TR1:{main_planner.scenario.scenario_id}:2020a.xml"))
    try:
        res = valid_solution(main_planner.scenario, main_planner.planning_problem_set, solution)
        logger.info(res)
    except Exception as e:
        logger.info(e)


if __name__ == '__main__':
    main()


def motion_planner_interactive(scenario_path: str, cfg: RLProjectConfig) -> Solution:
    cfg_obj = OmegaConf.to_object(cfg)

    main_planner = InteractivePlanner()
    paths = scenario_path.split('/')
    name_scenario = paths[-1]
    folder_scenarios = "/commonroad/scenarios"

    sumo_sim = main_planner.initialize(folder_scenarios, name_scenario)
    simulated_scenario, ego_vehicles = main_planner.process(sumo_sim, cfg_obj)

    planning_problem_set = main_planner.planning_problem_set
    vehicle_type = main_planner.vehicle_type
    vehicle_model = main_planner.vehicle_model
    cost_function = main_planner.cost_function
    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in planning_problem_set.planning_problem_dict
        state_initial = copy.deepcopy(planning_problem_set.planning_problem_dict[pp_id].initial_state)
        set_attributes_state_initial = set(state_initial.attributes)
        list_states_trajectory_full = [state_initial]

        # set missing attributes to correctly construct solution file
        for state in ego_vehicle.driven_trajectory.trajectory.state_list:
            set_attributes_state = set(state.attributes)

            set_attributes_in_state_extra = set_attributes_state.difference(set_attributes_state_initial)
            if set_attributes_in_state_extra:
                for attribute in set_attributes_in_state_extra:
                    setattr(state_initial, attribute, 0)

            set_attributes_in_state_initial_extra = set_attributes_state_initial.difference(set_attributes_state)
            if set_attributes_in_state_initial_extra:
                for attribute in set_attributes_in_state_initial_extra:
                    setattr(state, attribute, 0)

            list_states_trajectory_full.append(state)

        trajectory_full = Trajectory(initial_time_step=0, state_list=list_states_trajectory_full)
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=vehicle_type,
                                           vehicle_model=vehicle_model,
                                           cost_function=cost_function,
                                           trajectory=trajectory_full))

    solution = Solution(simulated_scenario.scenario_id, pps)

    return solution
