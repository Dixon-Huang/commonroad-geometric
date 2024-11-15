__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2024.1"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
from copy import deepcopy
import logging
from commonroad.common.file_reader import CommonRoadFileReader

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.utility.config import ReactivePlannerConfiguration

from commonroad_rp.utility.logger import initialize_logger


# *************************************
# Set Configurations
# *************************************
# filename = "ZAM_Tjunction-1_42_T-1.xml"
# filename = "/home/hya/commonroad-geometric/data/highd-sample/DEU_LocationALower-11_17_T-1.xml"
# filename = "/home/yanliang/commonroad-geometric/data/highd-sample/DEU_LocationALower-11_17_T-1.xml"
filename = "/home/yanliang/dataset/data/intersections_test/DEU_Lohmar-63_1_T-1.xml"
# filename = "/home/yanliang/dataset-converters/output/DEU_LocationALower-11_1_T-1.xml"

# Build config object
# config = ReactivePlannerConfiguration.load(f"reactive_planner_configurations/{filename[:-4]}.yaml", filename)
config = ReactivePlannerConfiguration.load(f"reactive_planner/reactive_planner_configurations/DEU_Test-1_1_T-1.yaml", filename)
config.update()

# initialize and get logger
initialize_logger(config)
logger = logging.getLogger("RP_LOGGER")


# *************************************
# Initialize Planner
# *************************************
# run route planner and add reference path to config
route_planner = RoutePlanner(config.scenario.lanelet_network, config.planning_problem)
route = route_planner.plan_routes().retrieve_first_route()

# initialize reactive planner
planner = ReactivePlanner(config)

# set reference path for curvilinear coordinate system
planner.set_reference_path(route.reference_path)

# **************************
# Run Planning
# **************************
# Add first state to recorded state and input list
planner.record_state_and_input(planner.x_0)

SAMPLING_ITERATION_IN_PLANNER = True

initial_state_1 = config.planning_problem.initial_state
initial_state_2 = planner.x_0

while not planner.goal_reached():
    current_count = len(planner.record_state_list) - 1

    # check if planning cycle or not
    plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
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
        if config.debug.show_plots or config.debug.save_plots:
            ego_vehicle = planner.convert_state_list_to_commonroad_object(optimal[0].state_list)
            sampled_trajectory_bundle = None
            if config.debug.draw_traj_set:
                sampled_trajectory_bundle = deepcopy(planner.stored_trajectories)
    else:
        # simulate scenario one step forward with planned trajectory
        sampled_trajectory_bundle = None

        # continue on optimal trajectory
        temp = current_count % config.planning.replanning_frequency

        # record state and input
        planner.record_state_and_input(optimal[0].state_list[1 + temp])

        # reset planner state for re-planning
        planner.reset(initial_state_cart=planner.record_state_list[-1],
                      initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                      collision_checker=planner.collision_checker, coordinate_system=planner.coordinate_system)

    print(f"current time step: {current_count}")

    # visualize the current time step of the simulation
    if config.debug.show_plots or config.debug.save_plots:
        visualize_planner_at_timestep(scenario=config.scenario, planning_problem=config.planning_problem,
                                      ego=ego_vehicle, traj_set=sampled_trajectory_bundle,
                                      ref_path=planner.reference_path, timestep=current_count, config=config)

# make gif
make_gif(config, range(0, planner.record_state_list[-1].time_step))


# **************************
# Evaluate results
# **************************
evaluate = True
if evaluate:
    cr_solution, feasibility_list = run_evaluation(planner.config, planner.record_state_list, planner.record_input_list)
