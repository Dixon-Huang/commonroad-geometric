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
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle, VehicleModel
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
from commonroad_geometric.common.config import Config
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.cost_function import DefaultCostFunctionFailSafe
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

    def check_state_again(self, current_scenario, ego_vehicle):
        # 路口规划器，交一部分由MCTS进行决策
        if self.lanelet_state == 3:
            ip = IntersectionPlanner(current_scenario, self.lanelet_route, ego_vehicle, self.lanelet_state)
            dis_ego2cp, _ = ip.desicion_making()
            if len(dis_ego2cp) == 0 or min(dis_ego2cp) > 150:
                self.lanelet_state = 4
                if not self.last_state == 4:  # 如果已经在4 不需要新的action
                    self.is_new_action_needed = 1  # 必须进入MCTS
        return

    # def check_state(self):
    #     """check if ego car is straight-going /incoming /in-intersection"""
    #     lanelet_id_ego = self.lanelet_ego
    #     ln = self.scenario.lanelet_network
    #     # find current lanelet
    #     potential_ego_lanelet_id_list = \
    #         self.scenario.lanelet_network.find_lanelet_by_position([self.ego_state.position])[0]
    #     for idx in potential_ego_lanelet_id_list:
    #         if idx in self.lanelet_route:
    #             lanelet_id_ego = idx
    #     self.lanelet_ego = lanelet_id_ego
    #     logger.info(f'current lanelet id: {self.lanelet_ego}')
    #
    #     for idx_inter, intersection in enumerate(ln.intersections):
    #         incomings = intersection.incomings
    #
    #         for idx_inc, incoming in enumerate(incomings):
    #             incoming_lanelets = list(incoming.incoming_lanelets)
    #             in_intersection_lanelets = list(incoming.successors_straight) + \
    #                                        list(incoming.successors_right) + list(incoming.successors_left)
    #
    #             for laneletid in incoming_lanelets:
    #                 if self.lanelet_ego == laneletid:
    #                     incoming_lanelet = ln.find_lanelet_by_id(self.lanelet_ego)
    #                     end_point = incoming_lanelet.center_vertices[-1]
    #                     distance = np.linalg.norm(end_point - self.ego_state.position)
    #                     if (distance < 10):
    #                         self.lanelet_state = 2  #即将进入intersection(距离intersection终点10m以内)
    #                         return self.lanelet_state
    #                     else:
    #                         self.lanelet_state = 1  #位于直路
    #                         return self.lanelet_state
    #
    #             for laneletid in in_intersection_lanelets:
    #                 if self.lanelet_ego == laneletid:
    #                     self.lanelet_state = 3  # in-intersection
    #                     return self.lanelet_state
    #
    #     if self.lanelet_state is None:
    #         self.lanelet_state = 1  # straighting-going
    #         return self.lanelet_state

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
                slip_angle=rp_state.slip_angle if hasattr(rp_state, 'slip_angle') else 0,
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

    def emergency_brake(self, factor=1.5):
        logger.info(f"==== perform emergency brake with factor: {factor} ====")
        self.is_new_action_needed = True

        next_state = copy.deepcopy(self.ego_state)
        # if next_state.steering_angle is None:
        next_state.steering_angle = 0.0
        next_state.yaw_rate = 0.0
        # a = -2.0
        vehicle_params = VehicleParameterMapping.from_vehicle_type(self.vehicle_type)
        # if self.ego_state.velocity > 20:
        #     a = - vehicle_params.longitudinal.a_max
        # elif self.ego_state.velocity > 10:
        #     a = - vehicle_params.longitudinal.a_max / 2
        # else:
        #     a = - vehicle_params.longitudinal.a_max / 3
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
            next_state.acceleration = a
        # ====== end of motion planner

        # update the ego vehicle with new trajectory with only 1 state for the current step
        next_state.time_step = 1
        next_state = self.convert_to_state_list([next_state])[0]
        return next_state

    def emergency_brake_with_reactive_planner(self, planner, sumo_ego_vehicle, dt, factor):
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
        delta_v = max(dt * planner.config.planning.time_steps_computation * a_max / factor, 5)

        # 计算期望速度
        current_velocity = sumo_ego_vehicle.current_state.velocity
        desired_velocity = current_velocity - delta_v if current_velocity > delta_v else 0

        # 设置planner的期望速度
        planner.set_desired_velocity(
            desired_velocity=desired_velocity,
            current_speed=planner.x_0.velocity,
            stopping=True
        )

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
        experiment_config = configure_experiment(Config(cfg_obj.experiment))
        model_config = configure_model(Config(cfg_obj.model))

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
        route = route_planner.plan_routes().retrieve_first_route()
        planner = ReactivePlanner(reactive_planner_config)
        planner.set_reference_path(route.reference_path)

        # **************************
        # Run Planning
        # **************************
        # Add first state to recorded state and input list
        planner.record_state_and_input(planner.x_0)

        logger.info("=" * 50 + " Start Simulation " + "=" * 50)
        for current_count in range(self.num_of_steps):
            logger.info(f"process: {current_count}/{self.num_of_steps}")

            current_scenario = sumo_sim.commonroad_scenario_at_time_step(current_count)
            for obstacle in current_scenario.dynamic_obstacles:
                obstacle.initial_state.time_step = current_count
                predicted_states = predict_constant_velocity(obstacle.initial_state, self.dt, planner.config.planning.time_steps_computation)
                trajectory = Trajectory(current_count, [obstacle.initial_state] + predicted_states)
                obstacle.prediction = TrajectoryPrediction(trajectory, shape=obstacle.obstacle_shape)
            planner.set_collision_checker(scenario=current_scenario)
            # reactive_planner_config.scenario = current_scenario
            planner.config.scenario = current_scenario

            logger.info(
                f"ego position: {sumo_ego_vehicle.current_state.position}, ego steering angle: {sumo_ego_vehicle.current_state.steering_angle}, ego velocity: {sumo_ego_vehicle.current_state.velocity}")
            logger.info(
                f"ego orientation: {sumo_ego_vehicle.current_state.orientation}, ego acceleration: {sumo_ego_vehicle.current_state.acceleration}, yaw rate: {sumo_ego_vehicle.current_state.yaw_rate}")

            # check if planning cycle or not
            plan_new_trajectory = current_count % reactive_planner_config.planning.replanning_frequency == 0
            if plan_new_trajectory:
                # new planning cycle -> plan a new optimal trajectory
                # planner.set_desired_velocity(current_speed=planner.x_0.velocity)
                planner.set_desired_curve_velocity(current_speed=planner.x_0.velocity)

                optimal = planner.plan()
                if not optimal:
                    break

                # record state and input
                planner.record_state_and_input(optimal[0].state_list[1])

                rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                next_states = rp_ego_vehicle.prediction.trajectory.state_list[1:]
                next_state = next_states[0]

                # reset planner state for re-planning
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              initial_state_curv=(optimal[2][1], optimal[3][1]),
                              collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

                # visualization: create ego Vehicle for planned trajectory and store sampled trajectory set
                if reactive_planner_config.debug.show_plots or reactive_planner_config.debug.save_plots:
                    ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                    sampled_trajectory_bundle = None
                    if reactive_planner_config.debug.draw_traj_set:
                        sampled_trajectory_bundle = copy.deepcopy(planner.stored_trajectories)
            else:
                # simulate scenario one step forward with planned trajectory
                sampled_trajectory_bundle = None

                # continue on optimal trajectory
                temp = current_count % reactive_planner_config.planning.replanning_frequency

                # record state and input
                planner.record_state_and_input(optimal[0].state_list[1 + temp])

                rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                next_states = rp_ego_vehicle.prediction.trajectory.state_list[1 + temp:]
                next_state = next_states[0]

                # reset planner state for re-planning
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                              collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

            # visualize the current time step of the simulation
            if reactive_planner_config.debug.show_plots or reactive_planner_config.debug.save_plots:
                visualize_planner_at_timestep(scenario=reactive_planner_config.scenario, planning_problem=reactive_planner_config.planning_problem,
                                              ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                              ref_path=planner.reference_path, timestep=current_count, config=reactive_planner_config)

            logger.info(
                f'next ego position: {next_state.position}, next ego steering angle: {next_state.steering_angle}, next ego velocity: {next_state.velocity}')
            logger.info(
                f'next ego orientation: {next_state.orientation}, next ego acceleration: {next_state.acceleration}, next yaw rate: {next_state.yaw_rate}')

            # ====== paste in simulations
            # ====== end of motion planner
            next_state.time_step = 1
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


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig):
    cfg_obj = OmegaConf.to_object(cfg)

    # folder_scenarios = os.path.abspath('/home/yanliang/commonroad-scenarios/scenarios/interactive/SUMO/')
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-interactive-scenarios/scenarios/tutorial")
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-scenarios/scenarios/interactive/hand-crafted")
    folder_scenarios = os.path.abspath("/home/yanliang/scenarios_phase")

    # name_scenario = "DEU_Frankfurt-73_2_I-1"
    # name_scenario = "DEU_Frankfurt-95_6_I-1"
    # name_scenario = "CHN_Sha-6_5_I-1-1"
    # name_scenario = "CHN_Sha-11_3_I-1-1" # 无法加载
    # name_scenario = "DEU_Frankfurt-152_8_I-1" # 时间不够
    # name_scenario = "ESP_Mad-2_1_I-1-1" # OK

    # name_scenario = "DEU_Cologne-63_5_I-1"
    # name_scenario = "DEU_Frankfurt-34_11_I-1"
    # name_scenario = "DEU_Aachen-2_1_I-1"
    # name_scenario = "DEU_Aachen-3_1_I-1"

    # name_scenario = "ZAM_Tjunction-1_270_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_547_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_517s_I-1-1" # 很难解
    # name_scenario = "DEU_Ffb-2_2_I-1-1"
    # name_scenario = "DEU_A9-2_1_I-1-1"
    # name_scenario = "ZAM_Zip-1_7_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_42_I-1-1"
    # name_scenario = "DEU_Muc-4_2_I-1-1" # 很难解，Route转弯很急
    # name_scenario = "ZAM_Tjunction-1_32_I-1-1"
    # name_scenario = "ZAM_Zip-1_69_I-1-1"
    # name_scenario = "ZAM_ACC-1_3_I-1-1"
    # name_scenario = "DEU_A9-1_2_I-1-1" # Simulation Failed 没有终点 就用普通reactive planner
    # name_scenario = "ZAM_Tjunction-1_258_I-1-1"  # 解不了
    # name_scenario = "ZAM_Tjunction-1_110_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_75_I-1-1"  # 修改了intersection信息
    # name_scenario = "DEU_Gar-1_2_I-1-1"

    # name_scenario = "DEU_Aachen-3_1_I-1" # OK
    # name_scenario = "DEU_Aachen-3_2_I-1" # OK
    # name_scenario = "DEU_Aachen-3_3_I-1" # OK
    name_scenario = "DEU_Aachen-3_7_I-1" # 跳变，待解决
    # name_scenario = "DEU_Aachen-3_11_I-1" # 最后一点会一直规划失败
    # name_scenario = "DEU_Cologne-27_9_I-1" # 地图太大
    # name_scenario = "DEU_Cologne-63_8_I-1" # OK
    # name_scenario = "DEU_Dresden-3_29_I-1" # OK
    # name_scenario = "DEU_Dresden-18_4_I-1" # OK
    # name_scenario = "DEU_Dresden-18_29_I-1" # OK
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
    # simulated scenarios
    path_scenarios_simulated = os.path.join(output_path, 'simulated_scenarios/')

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    trajectory._state_list = [ego_vehicle.initial_state] + trajectory.state_list

    # for state in trajectory.state_list:
    #     state.yaw_rate = 0

    # create mp4 animation
    create_video(simulated_scenario,
                 output_folder_path,
                 main_planner.planning_problem_set,
                 ego_vehicles,
                 True,
                 "_planner")

    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)
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

    # get feasible trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)

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
