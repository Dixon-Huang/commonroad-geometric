import sys
import matplotlib.pyplot as plt
import sys
import os
import copy
import hydra
import logging
import numpy as np
from torch import nn
from torch.optim import Adam
from dataclasses import dataclass
from omegaconf import OmegaConf
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.common.solution import VehicleType
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
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
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
                        incoming_lanelet = ln.find_lanelet_by_id(self.lanelet_ego)
                        end_point = incoming_lanelet.center_vertices[-1]
                        distance = np.linalg.norm(end_point - self.ego_state.position)
                        if(distance < 10):
                            self.lanelet_state = 2 #即将进入intersection(距离intersection终点10m以内)
                            return self.lanelet_state
                        else:
                            self.lanelet_state = 1 #位于直路
                            return self.lanelet_state

                for laneletid in in_intersection_lanelets:
                    if self.lanelet_ego == laneletid:
                        self.lanelet_state = 3  # in-intersection
                        return self.lanelet_state

        if self.lanelet_state is None:
            self.lanelet_state = 1  # straighting-going
            return self.lanelet_state

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
                jerk=rp_state.jerk if hasattr(rp_state, 'jerk') else 0,
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

    def visualize_trajectory(self, ego_vehicle, reactive_planner, config):
        if config.debug.show_plots or config.debug.save_plots:
            sampled_trajectory_bundle = None
            if config.debug.draw_traj_set:
                sampled_trajectory_bundle = copy.deepcopy(reactive_planner.stored_trajectories)
            visualize_planner_at_timestep(scenario=config.scenario,
                                          planning_problem=config.planning_problem,
                                          ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                          ref_path=reactive_planner.reference_path,
                                          timestep=ego_vehicle.prediction.trajectory.final_state.time_step,
                                          config=config)

    def initialize(self, folder_scenarios, name_scenario):
        self.vehicle_type = VehicleType.FORD_ESCORT
        self.vehicle_model = VehicleModel.KS
        self.cost_function = CostFunction.TR1
        self.vehicle = VehicleDynamics.KS(self.vehicle_type)

        interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

        conf = load_sumo_configuration(interactive_scenario_path)
        scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
        self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_file).open()

        scenario_wrapper = ScenarioWrapper()
        scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
        scenario_wrapper.initial_scenario = self.scenario

        self.num_of_steps = conf.simulation_steps
        sumo_sim = SumoSimulation()

        # initialize simulation
        # sumo_sim.initialize(conf, scenario_wrapper, None)
        sumo_sim.initialize(conf, scenario_wrapper, self.planning_problem_set)

        self.t_record = 0

        return sumo_sim

    def process(self, sumo_sim, cfg_obj):

        # generate ego vehicle
        sumo_ego_vehicles = sumo_sim.ego_vehicles

        # configure experiment and model
        experiment_config = configure_experiment(Config(cfg_obj.experiment))
        model_config = configure_model(Config(cfg_obj.model))
        simulation_options = experiment_config.simulation_options
        # extractor_factory = TrafficExtractorFactory(experiment_config.traffic_extraction_options)

        reactive_planner_config = _dict_to_params(
            OmegaConf.create(model_config.agent_kwargs['actor_options'].reactive_planner_options.as_dict()),
            ReactivePlannerConfiguration
        )

        dt = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step).dt
        self.dt = dt

        planning_problem_set = self.planning_problem_set
        planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[0]
        lanelets_of_goal_position = copy.deepcopy(planning_problem.goal.lanelets_of_goal_position)

        logger.info("=" * 50 + " Start Simulation " + "=" * 50)
        try:
            for step in range(self.num_of_steps):
                # for step in range(0, 10):

                logger.info(f"process: {step}/{self.num_of_steps}")
                current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
                sumo_ego_vehicle = list(sumo_ego_vehicles.values())[0]
                self.ego_state = sumo_ego_vehicle.current_state

                # initial positions do not match
                planning_problem.initial_state.position = copy.deepcopy(sumo_ego_vehicle.current_state.position)
                planning_problem.initial_state.orientation = copy.deepcopy(sumo_ego_vehicle.current_state.orientation)
                planning_problem.initial_state.velocity = copy.deepcopy(sumo_ego_vehicle.current_state.velocity)
                # ====== plug in your motion planner here
                # ====== paste in simulations

                self.t_record += 0.1
                if self.t_record > 1 and (self.last_semantic_action is None or self.last_semantic_action not in {1, 2}):
                    self.is_new_action_needed = True
                    logger.info('force to get a new action during straight-going')
                    self.t_record = 0

                # generate a CR planner
                next_state = self.planning(copy.deepcopy(current_scenario),
                                           planning_problem_set,
                                           sumo_ego_vehicle,
                                           sumo_sim.current_time_step,
                                           reactive_planner_config,
                                           simulation_options,
                                           # extractor_factory,
                                           dt,
                                           step,
                                           lanelets_of_goal_position)

                logger.info(f'next ego position: {next_state.position}, next ego velocity: {next_state.velocity}')

                # ====== paste in simulations
                # ====== end of motion planner
                next_state.time_step = 1
                next_state.steering_angle = 0.0
                trajectory_ego = [next_state]
                sumo_ego_vehicle.set_planned_trajectory(trajectory_ego)

                sumo_sim.simulate_step()
                logger.info("=" * 120)

                if self.is_reach_goal_region:
                    state_dict_keys = planning_problem.goal.state_list[0].__dict__.keys()
                    has_time_step = 'time_step' in state_dict_keys
                    has_orientation = 'orientation' in state_dict_keys

                    if not (has_time_step or has_orientation):
                        # 如果两个条件都没有
                        logger.info('goal region reached!')
                        break
                    elif has_time_step and not has_orientation:
                        # 只有time_step
                        if step * dt in planning_problem.goal.state_list[0].time_step:
                            logger.info('goal region reached!')
                            break
                    elif not has_time_step and has_orientation:
                        # 只有orientation
                        if sumo_ego_vehicle.current_state.orientation in planning_problem.goal.state_list[0].orientation:
                            logger.info('goal region reached!')
                            break
                    else:
                        # 两个条件都有
                        if (step * dt in planning_problem.goal.state_list[0].time_step and
                                sumo_ego_vehicle.current_state.orientation in planning_problem.goal.state_list[
                                    0].orientation):
                            logger.info('goal region reached!')
                            break
        finally:
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
            time_step,
            reactive_planner_config,
            simulation_options,
            # extractor_factory,
            dt,
            step,
            lanelets_of_goal_position
    ):
        logger.info(f"ego position: {sumo_ego_vehicle.current_state.position}, "
                    f"ego velocity: {sumo_ego_vehicle.current_state.velocity}")

        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

        action = self.last_action
        semantic_action = self.last_semantic_action
        # update action
        if not time_step == 0 and self.last_semantic_action in {1, 2}:
            action.T -= 0.1
            if action.T <= 0.5:
                self.is_new_action_needed = True

        # 判断是否到达目标
        try:
            is_goal = self.check_goal_state(sumo_ego_vehicle.current_state.position, lanelets_of_goal_position)
        except:
            is_goal = False

        if is_goal:
            logger.info('goal reached! braking!')

            if sumo_ego_vehicle.current_state.velocity == 0:
                return sumo_ego_vehicle.current_state

            # 到达目标区域，开始减速
            self.is_reach_goal_region = True

        current_count = step
        # check if planning cycle or not
        plan_new_trajectory = current_count % reactive_planner_config.planning.replanning_frequency == 0

        if plan_new_trajectory or self.is_new_action_needed:
            # run route planner and add reference path to config
            try:
                route_planner = RoutePlanner(scenario.lanelet_network,
                                             planning_problem)
                self.route = route_planner.plan_routes().retrieve_first_route()
                self.lanelet_route = self.route.lanelet_ids
            except:
                logger.info("Route planning failed")
            route = self.route

            # check if start to car-following
            front_veh_info = front_vehicle_info_extraction(self.scenario,
                                                           self.ego_state.position,
                                                           self.lanelet_route)

            num_time_steps = reactive_planner_config.planning.time_steps_computation
            for obstacle in scenario.obstacles:
                obstacle.initial_state.time_step = time_step
                predicted_states = predict_constant_velocity(obstacle.initial_state, dt, num_time_steps)
                trajectory = Trajectory(time_step, [obstacle.initial_state] + predicted_states)
                obstacle.prediction = TrajectoryPrediction(trajectory, shape=obstacle.obstacle_shape)

            # for i in range(time_step, time_step + num_time_steps):
            #     plt.figure(figsize=(25, 10))
            #     rnd = MPRenderer()
            #     rnd.draw_params.time_begin = i
            #     scenario.draw(rnd)
            #     planning_problem_set.draw(rnd)
            #     rnd.render()
            #     plt.show()

            # plt.figure(figsize=(25, 10))
            # rnd = MPRenderer()
            # rnd.draw_params.time_begin = time_step
            # scenario.draw(rnd)
            # planning_problem_set.draw(rnd)
            # rnd.render()
            # plt.show()

            # too close to front car, start to car-following
            if not (front_veh_info['dhw'] == -1 or time_step == 0 or self.ego_state.velocity == 0) and self.last_action:
                ttc = (front_veh_info['dhw'] - 5) / (self.ego_state.velocity - front_veh_info['v'])
                if 0 < ttc < 5 or front_veh_info['dhw'] < 20:
                    logger.info('ttc: ' + str(ttc))
                    logger.info('too close to front car, start to car-following')
                    action_temp = copy.deepcopy(action)
                    # IDM
                    s_t = 2 + max([0, self.ego_state.velocity * 1.5 - self.ego_state.velocity * (
                            self.ego_state.velocity - front_veh_info['v']) / 2 / (7 * 2) ** 0.5])
                    acc = max(7 * (1 - (self.ego_state.velocity /
                                        60 * 3.6) ** 5 - (s_t / (front_veh_info['dhw'] - 5)) ** 2), -7)
                    if acc > 5:
                        acc = 5
                    action_temp.T = 5
                    action_temp.v_end = self.ego_state.velocity + action_temp.T * acc
                    if action_temp.v_end < 0:
                        action_temp.v_end = 0
                    action_temp.delta_s = self.ego_state.velocity * 5 + 0.5 * acc * action_temp.T ** 2
                    action = action_temp

                    lattice_planner = Lattice_CRv3(scenario, sumo_ego_vehicle)
                    next_states_queue_temp, self.is_new_action_needed = lattice_planner.planner(action, semantic_action)
                    next_states = next_states_queue_temp[0: 4]
                    self.next_states_queue = self.convert_to_state_list(next_states)
                    next_state = self.next_states_queue.pop(0)
                    self.is_new_action_needed = True
                    return next_state

            # ego vehicle state
            ego_state = sumo_ego_vehicle.current_state
            ego_state.time_step = time_step
            ego_state.steering_angle = 0.0

            ego_vehicle = EgoVehicle(
                vehicle_model=self.vehicle_model,
                vehicle_type=self.vehicle_type,
                dt=dt,
            )
            # 设置ego初始状态
            ego_vehicle.reset(initial_state=ego_state)

            # 检查当前自车所在位置   1. 直路lanelet, 2. 即将进入intersection(距离intersection终点10m以内), 3. 位于intersection中
            self.last_state = copy.deepcopy(self.lanelet_state)
            self.lanelet_state = self.check_state()
            logger.info(f"intersection_state: {self.lanelet_state}")
            change_planner = True

            try:
                assert 'position' in planning_problem.goal.state_list[0].__dict__.keys()

                # 判断使用哪个planner
                if self.lanelet_state in {3} and self.is_reach_goal_region is False:
                    try:
                        ego_x, ego_y, velocity, yaw, change_planner = level_k_planner(
                            scenario, planning_problem_set, route, ego_vehicle)
                        if all(x is not None for x in [ego_x, ego_y, velocity, yaw]):
                            next_states = self.create_state_list(ego_x, ego_y, velocity, yaw)
                            self.next_states_queue = self.convert_to_state_list(next_states)
                            next_state = self.next_states_queue.pop(0)
                            self.is_new_action_needed = False
                            return next_state
                        else:
                            change_planner = True
                    except Exception as e:
                        logger.warning(f"*** level_k_planner failed ***\n"
                              f"failed reason: {e}")
                        try:
                            self.is_new_action_needed = True
                            ip = IntersectionPlanner(scenario, self.lanelet_route, sumo_ego_vehicle,
                                                     self.lanelet_state)
                            action = ip.planning(scenario)
                            semantic_action = 4
                            lattice_planner = Lattice_CRv3(scenario, sumo_ego_vehicle)
                            next_states, self.is_new_action_needed = lattice_planner.planner(action,
                                                                                                        semantic_action)
                            self.next_states_queue = self.convert_to_state_list(next_states)
                            next_state = self.next_states_queue.pop(0)
                            self.is_new_action_needed = False
                            return next_state

                            "update the last action info"
                            self.last_action = action
                            self.last_semantic_action = semantic_action

                            return next_state
                        except:
                            change_planner = True

                if self.lanelet_state in {1, 2} or change_planner is True or self.is_reach_goal_region:
                    # Initialize scenario simulation and traffic extractor
                    simulation = ScenarioSimulation(initial_scenario=scenario, options=simulation_options)
                    simulation.start()
                    # extractor = extractor_factory(simulation=simulation)
                    try:
                        simulation = simulation(
                            from_time_step=time_step,
                            ego_vehicle=ego_vehicle,
                        )
                    except:
                        simulation._final_time_step = time_step + 20
                        simulation = simulation(
                            from_time_step=time_step,
                            ego_vehicle=ego_vehicle,
                            force=True
                        )

                    simulation.spawn_ego_vehicle(ego_vehicle)
                    # extractor.reset_feature_computers()
                    ego_vehicle.ego_route = EgoRoute(
                        simulation=simulation,
                        planning_problem_set=planning_problem_set,
                    )

                    # 更新配置中的场景和规划问题
                    reactive_planner_config.update(scenario=scenario, planning_problem=planning_problem)

                    # *************************************
                    # Initialize Planner
                    # *************************************
                    # 初始化 reactive_planner
                    planner = ReactivePlanner(reactive_planner_config)

                    # set reference path for curvilinear coordinate system and desired velocity
                    planner.set_reference_path(route.reference_path)

                    if not self.is_reach_goal_region:
                        planner.set_cost_function(BaselineCostFunction(planner.desired_speed,
                                                                       desired_d=0.0,
                                                                       desired_s=planner.desired_lon_position,
                                                                       simulation=simulation,
                                                                       # traffic_extractor=extractor,
                                                                       ego_vehicle=ego_vehicle,
                                                                       ))

                    # **************************
                    # Run Planning
                    # **************************
                    # Add first state to recorded state and input list
                    planner.record_state_and_input(planner.x_0)

                    # new planning cycle -> plan a new optimal trajectory
                    if self.is_reach_goal_region:
                        planner.set_desired_velocity(desired_velocity=0, current_speed=planner.x_0.velocity,
                                                     stopping=True)
                    else:
                        planner.set_desired_velocity(current_speed=planner.x_0.velocity)
                        try:
                            self.is_reach_goal_region = planner.goal_reached()
                        except Exception as e:
                            logger.warning(f"*** goal_reached failed ***\n"
                                  f"failed reason: {e}")

                    # plan optimal trajectory
                    optimal = planner.plan()
                    if not optimal:  # 如果返回None或False
                        x_0 = planner.x_0
                        rp_failure_ego_vehicle = planner.convert_state_list_to_commonroad_object([x_0])
                        self.visualize_trajectory(rp_failure_ego_vehicle, planner, reactive_planner_config)
                        raise ValueError("Planning failed: optimal is None or False")

                    rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                    next_states = rp_ego_vehicle.prediction.trajectory.state_list[1:]
                    self.next_states_queue = self.convert_to_state_list(next_states)
                    self.visualize_trajectory(rp_ego_vehicle, planner, reactive_planner_config)

                    self.is_new_action_needed = False

            except Exception as e:
                logger.warning(f"*** reactive_planner failed *** \n"
                      f"failed reason: {e}")
                self.is_new_action_needed = True
                if self.is_reach_goal_region:
                    # 直接刹车
                    # next_state = brake(ego_vehicle.current_state, self.goal_info[1], self.goal_info[2])
                    action = brake(self.scenario, sumo_ego_vehicle)
                    self.is_new_action_needed = False
                    # update the last action info
                    self.last_action = action
                    # lattice planning
                    lattice_planner = Lattice_CRv3(self.scenario, sumo_ego_vehicle)
                    next_states, _ = lattice_planner.planner(action, 3)
                    self.next_states_queue = self.convert_to_state_list(next_states)
                    next_state = self.next_states_queue.pop(0)
                    return next_state

                if self.is_new_action_needed:
                    mcts_planner = MCTs_CR(scenario, planning_problem, self.lanelet_route, sumo_ego_vehicle)
                    semantic_action, action, self.goal_info = mcts_planner.planner(time_step)
                    self.is_new_action_needed = False
                else:
                    # update action
                    action.T -= 0.1
                    if action.T <= 0.5:
                        self.is_new_action_needed = True

                # lattice planning according to action
                lattice_planner = Lattice_CRv3(scenario, sumo_ego_vehicle)
                next_states, self.is_new_action_needed = lattice_planner.planner(action, semantic_action)
                self.next_states_queue = self.convert_to_state_list(next_states)
                next_state = self.next_states_queue.pop(0)

                # update the last action info
                self.last_action = action
                self.last_semantic_action = semantic_action
                return next_state

        next_state = self.next_states_queue.pop(0)
        return next_state


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig):
    cfg_obj = OmegaConf.to_object(cfg)

    # folder_scenarios = os.path.abspath('/home/yanliang/commonroad-scenarios/scenarios/interactive/SUMO/')
    folder_scenarios = os.path.abspath("/home/yanliang/commonroad-interactive-scenarios/scenarios/tutorial")
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-scenarios/scenarios/interactive/hand-crafted")
    # folder_scenarios = os.path.abspath("/home/yanliang/commonroad-scenarios/scenarios/interactive/hand-crafted")

    # name_scenario = "DEU_Frankfurt-73_2_I-1"
    # name_scenario = "DEU_Frankfurt-95_6_I-1"
    # name_scenario = "CHN_Sha-6_5_I-1-1"

    # name_scenario = "DEU_Cologne-63_5_I-1"
    # name_scenario = "DEU_Frankfurt-34_11_I-1"
    # name_scenario = "DEU_Dresden-3_20_I-1"
    name_scenario = "DEU_Aachen-2_1_I-1"

    # name_scenario = "ZAM_Tjunction-1_270_I-1-1"
    # name_scenario = "ZAM_Tjunction-1_517_I-1-1"
    # name_scenario = "DEU_Ffb-2_2_I-1-1"

    # name_scenario = "DEU_A9-2_1_I-1-1"

    main_planner = InteractivePlanner()

    sumo_sim = main_planner.initialize(folder_scenarios, name_scenario)
    simulated_scenario, ego_vehicles = main_planner.process(sumo_sim, cfg_obj)

    # path for outputting results
    output_path = '/home/xavier/cr-competition-outputs'

    # video
    output_folder_path = os.path.join(output_path, 'videos/')
    # solution
    path_solutions = os.path.join(output_path, 'solutions/')
    # simulated scenarios
    path_scenarios_simulated = os.path.join(output_path, 'simulated_scenarios/')

    # create mp4 animation
    create_video(simulated_scenario,
                 output_folder_path,
                 main_planner.planning_problem_set,
                 ego_vehicles,
                 True,
                 "_planner")

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    trajectory._state_list = [ego_vehicle.initial_state] + trajectory.state_list
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)

    logger.info('Feasible? {}'.format(feasible))
    recon_num = 0
    while not (feasible or recon_num >= 3):
        recon_num += 1
        # if not feasible. reconstruct the inputs
        initial_state = trajectory.state_list[0]
        vehicle = VehicleDynamics.KS(VehicleType.FORD_ESCORT)
        dt = 0.1
        reconstructed_states = [vehicle.convert_initial_state(initial_state)] + [
            vehicle.simulate_next_state(trajectory.state_list[idx], inp, dt)
            for idx, inp in enumerate(reconstructed_inputs.state_list)
        ]
        trajectory_reconstructed = Trajectory(initial_time_step=0, state_list=reconstructed_states)

        for i, state in enumerate(trajectory_reconstructed.state_list):
            ego_vehicle.driven_trajectory.trajectory.state_list[i] = state
        feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory_reconstructed,
                                                                                    main_planner.vehicle,
                                                                                    main_planner.dt)
        logger.info('after recon, Feasible? {}'.format(feasible))

    # saves trajectory to solution file
    save_solution(simulated_scenario, main_planner.planning_problem_set, ego_vehicles,
                  main_planner.vehicle_type,
                  main_planner.vehicle_model,
                  main_planner.cost_function,
                  path_solutions, overwrite=True)

    solution = CommonRoadSolutionReader.open(os.path.join(path_solutions,
                                                          f"solution_KS1:TR1:{name_scenario}:2020a.xml"))
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

    # get trajectory
    ego_vehicle = list(ego_vehicles.values())[0]
    trajectory = ego_vehicle.driven_trajectory.trajectory
    feasible, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory,
                                                                                main_planner.vehicle,
                                                                                main_planner.dt)
    # logger.info('Feasible? {}'.format(feasible))
    if not feasible:
        # if not feasible. reconstruct the inputs
        initial_state = trajectory.state_list[0]
        vehicle = VehicleDynamics.KS(VehicleType.FORD_ESCORT)
        dt = 0.1
        reconstructed_states = [vehicle.convert_initial_state(initial_state)] + [
            vehicle.simulate_next_state(trajectory.state_list[idx], inp, dt)
            for idx, inp in enumerate(reconstructed_inputs.state_list)
        ]
        trajectory_reconstructed = Trajectory(initial_time_step=0, state_list=reconstructed_states)
        # feasible_re, reconstructed_inputs = feasibility_checker.trajectory_feasibility(trajectory_reconstructed,
        #                                                                                main_planner.vehicle,
        #                                                                                main_planner.dt)
        for i, state in enumerate(trajectory_reconstructed.state_list):
            ego_vehicle.driven_trajectory.trajectory.state_list[i] = trajectory_reconstructed.state_list[i]
        # logger.info('after recon, Feasible? {}'.format(feasible_re))

    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in main_planner.planning_problem_set.planning_problem_dict
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=main_planner.vehicle_type,
                                           vehicle_model=main_planner.vehicle_model,
                                           cost_function=main_planner.cost_function,
                                           trajectory=ego_vehicle.driven_trajectory.trajectory))

    solution = Solution(simulated_scenario.scenario_id, pps)

    return solution
