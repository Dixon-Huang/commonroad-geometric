from commonroad.common.solution import VehicleType
from torch import nn
from torch.optim import Adam

from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import *
from commonroad_geometric.common.io_extensions.scenario import LaneletAssignmentStrategy
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.lanelet_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_lanelet import *
from commonroad_geometric.dataset.extraction.traffic.feature_computers.implementations.vehicle_to_vehicle import *
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions, TrafficFeatureComputerOptions
from commonroad_geometric.dataset.scenario.preprocessing.wrappers.chain_preprocessors import chain_preprocessors
from commonroad_geometric.learning.reinforcement import RLEnvironmentOptions
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment, RLExperimentConfig
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.learning.reinforcement.project.base_rl_project import BaseRLProject
from commonroad_geometric.learning.reinforcement.rewarder.reward_aggregator.implementations import SumRewardAggregator
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from projects.geometric_models.drivable_area.project import create_lanelet_graph_conversion_steps
from commonroad_geometric.learning.reinforcement.observer.implementations.ego_enhanced_graph_observer import EgoEnhancedGraphObserver
from commonroad_geometric.learning.reinforcement.training.rl_trainer import RLModelConfig
from commonroad_geometric.rendering.plugins.cameras.follow_vehicle_camera import FollowVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.ego_vehicle_camera import EgoVehicleCamera
from commonroad_geometric.rendering.plugins.implementations import *
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRendererOptions
# from commonroad_geometric.simulation.ego_simulation.control_space.implementations.pid_control_space import PIDControlOptions, PIDControlSpace
from commonroad_geometric.simulation.ego_simulation.control_space.implementations.pid_control_space_modified import PIDControlOptions, PIDControlSpace
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import VehicleModel
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulationOptions
from commonroad_geometric.simulation.ego_simulation.respawning.implementations import (RandomRespawner,
                                                                                       RandomRespawnerOptions)
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)
from commonroad_geometric.simulation.interfaces.interactive.sumo_simulation import (SumoSimulation,
                                                                                    SumoSimulationOptions)
from commonroad_geometric.simulation.interfaces.static.unpopulated_simulation import UnpopulatedSimulation
from projects.graph_rl_agents.v2v_policy.feature_extractor import VehicleGraphFeatureExtractor

from commonroad_rp.utility.config import ReactivePlannerConfiguration
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.reactive_planner_agent import ReactivePlannerAgent
from projects.graph_rl_agents.predictive_traffic_rule_compliance.simulation.reactive_planner_control_space import ReactivePlannerControlSpace, ReactivePlannerControlOptions  # import custom control space
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.actor.reactive_planner_actor import ReactivePlannerActor, ReactivePlannerActorOptions

# Robustness related imports
from projects.graph_rl_agents.predictive_traffic_rule_compliance.data.graph_features.rl_project.robustness_penalty import (
    G1RobustnessPenaltyRewardComputer,
    G2RobustnessPenaltyRewardComputer,
    G3RobustnessPenaltyRewardComputer,
    G4RobustnessPenaltyRewardComputer,
    I1RobustnessPenaltyRewardComputer,
    I2RobustnessPenaltyRewardComputer,
    I3RobustnessPenaltyRewardComputer,
    I4RobustnessPenaltyRewardComputer,
    I5RobustnessPenaltyRewardComputer
)
from projects.graph_rl_agents.predictive_traffic_rule_compliance.data.graph_features.robustness_feature_computer import (
    RuleRobustnessFeatureComputer,
    PredicateRobustnessFeatureComputer,
)


def create_rewarders(stage):
    if stage == 1:
        rewarders = [
            G4RobustnessPenaltyRewardComputer(weight=1),
            # CollisionPenaltyRewardComputer(
            #     penalty=-1.5,
            # ),
        ]
    elif stage == 2:
        rewarders = [
            G1RobustnessPenaltyRewardComputer(weight=1),
            G2RobustnessPenaltyRewardComputer(weight=1),
        ]
    elif stage == 3:
        rewarders = [
            G1RobustnessPenaltyRewardComputer(weight=1),
            G2RobustnessPenaltyRewardComputer(weight=1),
            G3RobustnessPenaltyRewardComputer(weight=1),
        ]

    return rewarders

def create_scenario_filterers():
    return []

def create_scenario_preprocessors():
    scenario_preprocessors = [
        SegmentLaneletsPreprocessor(100.0),
        ComputeVehicleVelocitiesPreprocessor()
    ]
    return scenario_preprocessors

def create_termination_criteria():
    termination_criteria = [
        OffroadCriterion(),
        CollisionCriterion(),
        ReachedGoalCriterion(),
        OvershotGoalCriterion(),
        TrafficJamCriterion(),
    ]
    return termination_criteria

RENDERER_OPTIONS = [
    TrafficSceneRendererOptions(
        camera=EgoVehicleCamera(view_range=200.0),
        plugins=[
            RenderLaneletNetworkPlugin(lanelet_linewidth=0.64),
            RenderPlanningProblemSetPlugin(
                render_trajectory=True,
                render_start_waypoints=True,
                render_goal_waypoints=True,
                render_look_ahead_point=True
            ),
            RenderTrafficGraphPlugin(),
            RenderEgoVehiclePlugin(
                render_trail=False,
            ),
            RenderObstaclePlugin(
                from_graph=False,
            ),
            RenderOverlayPlugin()
        ],
    ),
]

V_FEATURE_COMPUTERS = [
    ft_veh_state,
    GoalAlignmentComputer(
        include_goal_distance_longitudinal=True,
        include_goal_distance_lateral=True,
        include_goal_distance=True,
        include_lane_changes_required=True,
        logarithmic=True
    ),
    YawRateFeatureComputer(),
    VehicleLaneletPoseFeatureComputer(
        include_longitudinal_abs=False,
        include_longitudinal_rel=False,
        include_lateral_left=True,
        include_lateral_right=True,
        include_lateral_error=False,
        include_heading_error=True,
        update_exact_interval=10
    ),
    EgoFramePoseFeatureComputer(),
    NumLaneletAssignmentsFeatureComputer(),
    DistanceToRoadBoundariesFeatureComputer(),
    # RuleRobustnessFeatureComputer(),
]
L_FEATURE_COMPUTERS = [
    LaneletGeometryFeatureComputer(),
]
L2L_FEATURE_COMPUTERS = [
    LaneletConnectionGeometryFeatureComputer(),
]
V2V_FEATURE_COMPUTERS = [
    ClosenessFeatureComputer(),
    TimeToCollisionFeatureComputer(),
    ft_rel_state_ego,
    # PredicateRobustnessFeatureComputer(g1_rule=True, g2_rule=True, g3_rule=True),
    # PredicateRobustnessFeatureComputer(),
]
V2L_FEATURE_COMPUTERS = [
    VehicleLaneletPoseEdgeFeatureComputer(update_exact_interval=1)
]

class PredictiveTrafficRuleComplianceProject(BaseRLProject):

    def __init__(self, cfg: dict, stage: int):
        self.stage = stage
        super().__init__(cfg)

    def configure_experiment(self, cfg: dict) -> RLExperimentConfig:
        # reactive_planner_config = cfg["reactive_planner"]
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
            # simulation_cls=SumoSimulation if cfg["enable_traffic"] else UnpopulatedSimulation,
            # simulation_options=SumoSimulationOptions(
            #     presimulation_steps=cfg["sumo_simulation"]["presimulation_steps"],
            #     p_wants_lane_change=cfg["sumo_simulation"]["p_wants_lane_change"],
            # ),

            # control_space_cls=ReactivePlannerControlSpace,  # use custom control space
            # control_space_options=ReactivePlannerControlOptions(reactive_planner_config),  # pass reactive planner config

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
            rewarder=SumRewardAggregator(create_rewarders(self.stage)),
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

    def configure_model(self, cfg: dict, experiment: RLExperiment) -> RLModelConfig:
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
