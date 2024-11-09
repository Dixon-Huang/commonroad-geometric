import matplotlib.pyplot as plt
import sys
import os
import hydra
import logging
import numpy as np
from torch import nn
from torch.optim import Adam
from dataclasses import dataclass
from copy import deepcopy
from omegaconf import OmegaConf
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

sys.path.insert(0, os.getcwd())


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
            vehicle_type=VehicleType.BMW_320i
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


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig):
    # generate path of the file to be opened
    # file_path = "/home/yanliang/dataset/data/highd-sample/DEU_LocationALower-11_13_T-1.xml"  # highway
    # file_path = "/home/yanliang/commonroad-scenarios/scenarios/recorded/NGSIM/US101/USA_US101-3_1_T-1.xml" # USA_highway
    # file_path = "/home/yanliang/dataset/data/t_junction_test/ZAM_Tjunction-1_4_T-1.xml"  # T-junction
    # file_path = "/home/yanliang/commonroad-geometric/projects/graph_rl_agents/predictive_traffic_rule_compliance/ZAM_Tjunction-1_75_T-1.xml" # T-junction
    file_path = "/home/yanliang/commonroad-scenarios/scenarios/recorded/NGSIM/Peachtree/USA_Peach-4_7_T-1.xml"  # crossroad
    # file_path = "/home/yanliang/dataset/data/interaction_sample/BGR_Interaction-1_8_T-1.xml"  # crossroad

    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    cfg_obj = OmegaConf.to_object(cfg)
    experiment_config = configure_experiment(Config(cfg_obj.experiment))
    model_config = configure_model(Config(cfg_obj.model))

    simulation_options = experiment_config.simulation_options
    extractor_factory = TrafficExtractorFactory(experiment_config.traffic_extraction_options)

    reactive_planner_config = _dict_to_params(
        OmegaConf.create(model_config.agent_kwargs['actor_options'].reactive_planner_options.as_dict()),
        ReactivePlannerConfiguration
    )

    # Initialize scenario simulation and traffic extractor
    simulation = ScenarioSimulation(initial_scenario=scenario, options=simulation_options)
    simulation.start()
    extractor = extractor_factory(simulation=simulation)
    ego_vehicle = EgoVehicle(
        vehicle_model=experiment_config.ego_vehicle_simulation_options.vehicle_model,
        vehicle_type=experiment_config.ego_vehicle_simulation_options.vehicle_type,
        dt=simulation.dt,
        ego_route=None,
    )
    simulation = simulation(
        from_time_step=0,
        ego_vehicle=ego_vehicle,
    )

    # 获取当前的场景和规划问题
    scenario = simulation.current_scenario
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    # end_velocity = planning_problem.goal.state_list[0].velocity.end

    # # 画图
    # # plot the planning problem and the scenario for the fifth time step
    # plt.figure(figsize=(25, 10))
    # rnd = MPRenderer()
    # rnd.draw_params.time_begin = 5
    # scenario.draw(rnd)
    # planning_problem_set.draw(rnd)
    # rnd.render()
    # plt.show()

    # visualize_scenario(scenario, planning_problem_set)

    # 设置ego初始状态
    ego_state = planning_problem.initial_state
    ego_state.time_step = simulation.current_time_step
    ego_state.steering_angle = 0.0
    ego_vehicle.reset(initial_state=ego_state)
    simulation.spawn_ego_vehicle(ego_vehicle)
    extractor.reset_feature_computers()
    ego_vehicle.ego_route = EgoRoute(
        simulation=simulation,
        planning_problem_set=planning_problem_set,
    )

    # 更新配置中的场景和规划问题
    reactive_planner_config.update(scenario=scenario, planning_problem=planning_problem)

    # *************************************
    # Initialize Planner
    # *************************************
    # run route planner and add reference path to config
    route_planner = RoutePlanner(reactive_planner_config.scenario.lanelet_network,
                                 reactive_planner_config.planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()

    # 初始化 reactive_planner
    planner = ReactivePlanner(reactive_planner_config)

    # set reference path for curvilinear coordinate system and desired velocity
    planner.set_reference_path(route.reference_path)

    if not hasattr(planning_problem.goal.state_list[0], 'velocity'):
        route_length = max(route.path_length_per_point)
        desired_velocity = route_length / ((planning_problem.goal.state_list[0].time_step.end
                                            + planning_problem.goal.state_list[0].time_step.start) / 2
                                           - ego_state.time_step)

    planner.set_cost_function(BaselineCostFunction(planner.desired_speed,
                                                   desired_d=0.0,
                                                   desired_s=planner.desired_lon_position,
                                                   simulation=simulation,
                                                   traffic_extractor=extractor,
                                                   ego_vehicle=ego_vehicle,
                                                   ))

    # **************************
    # Run Planning
    # **************************
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)

    SAMPLING_ITERATION_IN_PLANNER = True

    while not planner.goal_reached():

        # 检查当前自车所在位置   1. 直路lanelet, 2. 即将进入intersection(距离intersection终点10m以内), 3. 位于intersection中
        intersection_state = check_state(scenario, ego_state, route)
        print(f"intersection_state: {intersection_state}")
        change_planner = False

        # if intersection_state == 3:
        #     pass

        if intersection_state in {2, 3}:
            # ego_states, change_planner = level_k_planner(scenario, planning_problem_set, route)
            # next_state = InitialState(
            #     position=ego_state[0],
            #     velocity=ego_state[1],
            #     yew_rate=ego_state[2],
            #     acceleration=ego_state[3],
            #     orientation=ego_state[4],
            #     time_step=planner.record_state_list[-1].time_step,
            #     slip_angle=0
            # )
            # if change_planner:
            #     current_count = 0

            change_planner = True

        if intersection_state in {1} or change_planner is True:
            current_count = len(planner.record_state_list) - 1

            # check if planning cycle or not
            plan_new_trajectory = current_count % reactive_planner_config.planning.replanning_frequency == 0
            if plan_new_trajectory:
                # new planning cycle -> plan a new optimal trajectory
                planner.set_desired_velocity(current_speed=planner.x_0.velocity)
                if SAMPLING_ITERATION_IN_PLANNER:
                    optimal = planner.plan()
                else:
                    optimal = None
                    i = 1
                    while optimal is None and i <= planner.sampling_level:
                        optimal = planner.plan(i)
                        next_state = optimal[0].state_list

                if not optimal:
                    break

                # record state and input
                planner.record_state_and_input(optimal[0].state_list[1])

                # reset planner state for re-planning
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              initial_state_curv=(optimal[2][1], optimal[3][1]),
                              collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

                # visualization: create ego Vehicle for planned trajectory and store sampled trajectory set
                if reactive_planner_config.debug.show_plots or reactive_planner_config.debug.save_plots:
                    rp_ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
                    sampled_trajectory_bundle = None
                    if reactive_planner_config.debug.draw_traj_set:
                        sampled_trajectory_bundle = deepcopy(planner.stored_trajectories)
            else:
                # simulate scenario one step forward with planned trajectory
                sampled_trajectory_bundle = None

                # continue on optimal trajectory
                temp = current_count % reactive_planner_config.planning.replanning_frequency

                # record state and input
                planner.record_state_and_input(optimal[0].state_list[1 + temp])

                # reset planner state for re-planning
                planner.reset(initial_state_cart=planner.record_state_list[-1],
                              initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                              collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

            print(f"current time step: {current_count}")

            # visualize the current time step of the simulation
            if reactive_planner_config.debug.show_plots or reactive_planner_config.debug.save_plots:
                visualize_planner_at_timestep(scenario=reactive_planner_config.scenario,
                                              planning_problem=reactive_planner_config.planning_problem,
                                              ego=rp_ego_vehicle, traj_set=sampled_trajectory_bundle,
                                              ref_path=planner.reference_path, timestep=current_count,
                                              config=reactive_planner_config)

            next_state = InitialState(
                position=planner.record_state_list[-1].position,
                velocity=planner.record_state_list[-1].velocity,
                orientation=planner.record_state_list[-1].orientation,
                yaw_rate=planner.record_state_list[-1].yaw_rate,
                acceleration=planner.record_state_list[-1].acceleration,
                time_step=planner.record_state_list[-1].time_step,
                slip_angle=0
            )

            print(f"车辆速度：{planner.record_state_list[-1].velocity}")

            ego_vehicle.set_next_state(next_state)
            next(simulation)

        print("=" * 120)

    # make gif
    make_gif(reactive_planner_config, range(0, planner.record_state_list[-1].time_step))

    # **************************
    # Evaluate results
    # **************************
    evaluate = True
    if evaluate:
        cr_solution, feasibility_list = run_evaluation(planner.config, planner.record_state_list,
                                                       planner.record_input_list)

    def visualize_trajectory(ego_vehicle, reactive_planner):
        sampled_trajectory_bundle = None
        sampled_trajectory_bundle = deepcopy(reactive_planner.stored_trajectories)

        visualize_planner_at_timestep(scenario=reactive_planner_config.scenario,
                                      planning_problem=reactive_planner_config.planning_problem,
                                      ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                      ref_path=reactive_planner.reference_path,
                                      timestep=ego_vehicle.prediction.trajectory.final_state.time_step,
                                      config=reactive_planner_config)


if __name__ == "__main__":
    main()
