import os
from commonroad.common.file_reader import CommonRoadFileReader
import os, sys
from commonroad.common.file_reader import CommonRoadFileReader
from sumo_simulation.simulations import load_sumo_configuration
# from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.interface.sumo_simulation import SumoSimulation
import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer

folder_scenarios = "/home/yanliang/commonroad-scenarios/scenarios/interactive/SUMO/"
# name_scenario = "DEU_Frankfurt-4_3_I-1"  # 交叉口测试场景
name_scenario = "DEU_Frankfurt-95_6_I-1"  # 直道测试场景
# name_scenario = "DEU_Frankfurt-73_6_I-1"
interactive_scenario_path = os.path.join(folder_scenarios, name_scenario)

conf = load_sumo_configuration(interactive_scenario_path)
scenario_file = os.path.join(interactive_scenario_path, f"{name_scenario}.cr.xml")
scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

# plot the planning problem and the scenario for the fifth time step
plt.figure(figsize=(25, 10))
rnd = MPRenderer()
rnd.draw_params.time_begin = 5
scenario.draw(rnd)
planning_problem_set.draw(rnd)
rnd.render()
plt.show()