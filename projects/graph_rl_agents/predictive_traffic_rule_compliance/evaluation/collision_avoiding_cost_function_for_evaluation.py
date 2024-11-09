from abc import ABC, abstractmethod
import logging
from typing import Optional
from commonroad_rp.cost_function import CostFunction
from commonroad.scenario.state import State, InitialState, CustomState, InputState
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
# from commonroad.scenario.trajectory import Trajectory, State
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad_rp.utility.config import ReactivePlannerConfiguration, VehicleConfiguration

import numpy as np
import copy

import math
import commonroad_rp.trajectories

import torch

logger = logging.getLogger(__name__)


class CollisionAvoidingCostFunction(CostFunction):
    """
    Cost function integrating value function as the robustness measure
    """

    def __init__(self,
                 desired_speed: Optional[float] = None,
                 desired_d: float = 0.0,
                 desired_s: Optional[float] = None,
                 simulation=None,
                 ego_vehicle=None
                 ) -> object:
        super(CollisionAvoidingCostFunction, self).__init__()
        # target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s
        self.simulation = simulation
        self.scenario = simulation.current_scenario

        # weights
        self.w_a = 5  # acceleration weight
        self.w_r = 5  # robustness weight

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        try:
            costs = 0.0

            # Other cost calculations
            # acceleration costs
            costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)
            # velocity costs
            if self.desired_speed is not None:
                costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                         (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                         (100 * (trajectory.cartesian.v[
                                     int(len(trajectory.cartesian.v) / 2)] - self.desired_speed) ** 2)
                # print(trajectory.cartesian.x)
            if self.desired_s is not None:
                costs += np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                         (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2
            # distance costs
            costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                     (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
            # orientation costs
            costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                    5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

            # calculate the distance from other vehicles or obstacles
            distances = 0.0
            # print("The initial time step of trajectory is:", trajectory.cartesian.)
            # for static obstacles:
            for ob in self.scenario.static_obstacles:
                # get x, y position of static obstacles
                sob_x = ob.initial_state.position[0]
                sob_y = ob.initial_state.position[1]
                # calculate the average distance of all trajectory points
                sum_dis = 0.0
                avg_dis = 0.0
                for i in range(len(trajectory.cartesian.x)):
                    dis_square = (trajectory.cartesian.x[i] - sob_x) ** 2 + (trajectory.cartesian.y[i] - sob_y) ** 2
                    if dis_square <= 4:
                        sum_dis += dis_square
                avg_dis = sum_dis / len(trajectory.cartesian.x)
                distances += avg_dis
            costs -= 10 * distances

            # 得到所有路径点到各个静态障碍物之间的平均距离之和
            # 接下来，计算动态障碍物的距离：路径点0与各个动态障碍物在时间点0的平均距离+路径点1与各个动态障碍物在时间点1的平均距离+……
            # 首先还是要得到各个动态障碍物的state
            distance_from_dynamic_ob = 0.0
            initial_time_step = self.simulation.current_time_step
            for ob in self.scenario.dynamic_obstacles:
                if ob.obstacle_id == -1: # ignore the ego vehicle
                    continue
                sum_distance_from_ob = 0.0
                for current_time_step in range(len(trajectory.cartesian.x)):
                    ob_states = state_at_time(ob, current_time_step + initial_time_step, assume_valid=False)
                    # only consider the already presented obstacles
                    # if ob_states is None, then the obstacle has not jet shown up, or has already faded away.
                    if ob_states:
                        dis_at_current_time_step = (ob_states.position[0] - trajectory.cartesian.x[
                            current_time_step]) ** 2 + \
                                                   (ob_states.position[1] - trajectory.cartesian.y[
                                                       current_time_step]) ** 2
                        # only pay attention to nearby obstacles
                        if dis_at_current_time_step <= 1000:
                            sum_distance_from_ob += dis_at_current_time_step
                # after that, we have get the sum distance from a certain obstacel during the whole time
                # now we need to sum them up to get the total distance from all obstacels
                distance_from_dynamic_ob += sum_distance_from_ob
            # normalize the total distance with time steps, and use it as a cost:
            costs -= 10 * distance_from_dynamic_ob / len(trajectory.cartesian.x)
            # print("costs is: ", costs, "distances is: ", distances)

        except Exception as e:
            logger.error(f"Error in cost function calculation: {e}")

        # finally:
        #    # 无论评估是否成功，都恢复到原始状态
        #    self.ego_vehicle_simulation.restore_state(saved_state)
        #    pass

        logger.debug(f"Costs Type: {type(costs)}")

        if isinstance(costs, torch.Tensor):
            return costs.item()
        else:
            return float(costs)