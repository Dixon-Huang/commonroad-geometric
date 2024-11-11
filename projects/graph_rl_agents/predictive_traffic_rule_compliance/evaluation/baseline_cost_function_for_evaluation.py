import copy
import logging
from typing import Optional
from commonroad_rp.cost_function import CostFunction
from commonroad_geometric.common.io_extensions.obstacle import state_at_time
from commonroad.scenario.state import InitialState
import numpy as np
import math
import commonroad_rp.trajectories
from crmonitor.evaluation.evaluation import RuleEvaluator

logger = logging.getLogger(__name__)


class BaselineCostFunction(CostFunction):
    """
    Cost function using rule robustness directly as the cost, evaluated at specific points.
    Evaluates multiple rules and sums their robustness values.
    """

    def __init__(self, desired_speed: Optional[float] = None,
                 desired_d: float = 0.0,
                 desired_s: Optional[float] = None,
                 simulation=None,
                 traffic_extractor=None,
                 ego_vehicle=None,
                 speed_limitation=None,
                 robustness_types: Optional[list] = None,
                 ) -> None:
        super(BaselineCostFunction, self).__init__()
        # Target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s
        self.simulation = simulation
        self.initial_time_step = copy.deepcopy(self.simulation.current_time_step)
        self.scenario = simulation.current_scenario
        self.ego_vehicle = ego_vehicle
        self.traffic_extractor = traffic_extractor
        self.speed_limitation = speed_limitation
        # Weights
        self.w_a = 5  # Acceleration weight
        self.w_r = 5  # Robustness weight
        self.w_low_speed = 3  # Low speed weight
        self.min_desired_speed = self.speed_limitation if self.speed_limitation else 10  # Minimum desired speed
        # self.base_d_weight = 0.25  # 基础距离权重
        # self.final_d_weight = 20.0  # 终点距离权重
        # self.distance_threshold = 30.0  # 开始增加权重的距离阈值（米）
        # self.exp_factor = 2.0  # 指数增长因子

        # self.saved_ego_vehicle_world_state = copy.deepcopy(self.simulation.world_state.vehicle_by_id(-1))

        # List of robustness types (rules)
        if robustness_types is None:
            # Default to all the rules you provided
            self.robustness_types = [
                'R_G1',
                # 'R_G2',
                # 'R_G3',
                # 'R_G4',
                # 'R_I1',
                # 'R_I2',
                # 'R_I3',
                # 'R_I4',
                # 'R_I5'
            ]
        else:
            self.robustness_types = robustness_types

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):

        costs = 0.0

        # Compute rule robustness at specific points along the trajectory
        robustness_list, indices, time_duration = self._calculate_robustness(trajectory)
        robustness_array = np.array(robustness_list)

        # Since we are minimizing cost, and higher robustness is better,
        # we subtract the robustness from the cost (or equivalently, add negative robustness)
        # Multiply by weight and scaling factor
        costs -= 1e2 * (np.sum(5 * robustness_array) / len(self.robustness_types) + \
                        (50 * robustness_array[-1]) + \
                        (100 * robustness_array[int(len(robustness_array) / 2)]))

        # Other cost calculations
        # Acceleration costs
        costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)

        # Velocity costs
        if self.desired_speed is not None:
            costs += 3 * np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                     (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                     (100 * (trajectory.cartesian.v[
                                 int(len(trajectory.cartesian.v) / 2)] - self.desired_speed) ** 2)
        if self.desired_s is not None:
            costs += 50 * 5 * np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                     (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2

        # Distance costs
        costs += 50 * 5 * np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                 (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2

        # Orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        # low speed costs
        # try:
        #     min_speed = np.minimum(self.min_desired_speed, self.desired_speed)
        # except:
        #     min_speed = self.min_desired_speed
        min_speed = self.min_desired_speed - 1
        speed_diff = np.maximum(0, min_speed - trajectory.cartesian.v)
        costs += self.w_low_speed * np.sum(np.exp(speed_diff) - 1)
        # costs += self.w_low_speed * np.sum(speed_diff ** 2)

        epsilon = 1e-6  # 防止除以零
        beta = 2e2  # 静态障碍物权重
        gamma = beta  # 动态障碍物权重

        # 使用反比例函数计算与静态障碍物的距离成本
        distances = 0.0
        for ob in self.scenario.static_obstacles:
            sob_x = ob.initial_state.position[0]
            sob_y = ob.initial_state.position[1]
            for i in range(len(trajectory.cartesian.x)):
                distance = np.sqrt(
                    (trajectory.cartesian.x[i] - sob_x) ** 2 +
                    (trajectory.cartesian.y[i] - sob_y) ** 2
                )
                cost = 1 / (distance + epsilon)
                distances += cost
        costs += beta * distances  # 距离障碍物越近，成本越高

        # 使用反比例函数计算与动态障碍物的距离成本
        distance_from_dynamic_ob = 0.0
        for ob in self.scenario.dynamic_obstacles:
            if ob.obstacle_id == -1:  # 忽略自车
                continue
            for current_time_step in range(len(trajectory.cartesian.x)):
                ob_states = state_at_time(ob, current_time_step + self.initial_time_step, assume_valid=False)
                if ob_states:
                    distance = np.sqrt(
                        (ob_states.position[0] - trajectory.cartesian.x[current_time_step]) ** 2 +
                        (ob_states.position[1] - trajectory.cartesian.y[current_time_step]) ** 2
                    )
                    cost = 1 / (distance + epsilon)
                    distance_from_dynamic_ob += cost
        costs += gamma * distance_from_dynamic_ob  # 距离障碍物越近，成本越高

        # Restore the original state
        self.ego_vehicle.reset_state_list(time_duration=time_duration)
        self.simulation.close()
        self.simulation.start()
        self.simulation = self.simulation(
            from_time_step=self.ego_vehicle.current_time_step,
            ego_vehicle=self.ego_vehicle,
        )
        self.simulation.lifecycle.run_ego()
        # self.traffic_extractor._simulation = self.simulation
        # 重置world_state的ego_vehicle状态
        remove_future_states(self.simulation.world_state.vehicle_by_id(-1), self.initial_time_step)

        return costs

    def _calculate_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        """
        Compute the sum of rule robustness at specific points along the trajectory.
        """
        try:
            ego_vehicle = self.ego_vehicle
            robustness_list = []
            # Initialize RuleEvaluator with all the rules
            ego_vehicle_world_state = self.simulation.world_state.vehicle_by_id(-1)
            rule_evaluators = []
            # ego_vehicle_world_state.end_time = self.simulation.current_time_step + len(trajectory.cartesian.x) - 1
            for rule in self.robustness_types:
                rule_evaluator = RuleEvaluator.create_from_config(
                    self.simulation.world_state,
                    ego_vehicle_world_state,
                    rule
                )
                rule_evaluators.append(rule_evaluator)

            num_points = len(trajectory.cartesian.x)
            indices = self._get_evaluation_indices(num_points)
            indices_set = set(indices)  # For faster lookup
            # initial_time_step = copy.deepcopy(self.simulation.current_time_step)

            # Dummy run to initialize the simulation
            for evaluator in rule_evaluators:
                evaluator._last_evaluation_time_step += 1

            for i in range(len(trajectory.cartesian.x) - 1):
                # Transform coordinates
                position_rear = np.array([trajectory.cartesian.x[i + 1], trajectory.cartesian.y[i + 1]])
                orientation = trajectory.cartesian.theta[i + 1]
                position_center = position_rear + ego_vehicle.parameters.b * np.array(
                    [math.cos(orientation), math.sin(orientation)])
                time_step = self.initial_time_step + 1 + i

                state = InitialState(
                    position=position_center,
                    velocity=trajectory.cartesian.v[i + 1],
                    acceleration=trajectory.cartesian.a[i + 1],
                    orientation=orientation,
                    time_step=time_step,
                    yaw_rate=0.0,
                    slip_angle=0.0
                )

                ego_vehicle.set_next_state(state)
                next(self.simulation)

                # if i + 1 in indices_set:
                total_robustness = 0.0
                # ego_vehicle_world_state.end_time = self.simulation.current_time_step
                for evaluator in rule_evaluators:
                    try:
                        robustness_series = evaluator.evaluate()
                        robustness = robustness_series[-1]
                        # 如果robustness为inf，说明规则不适用，不计入总robustness
                        if robustness == float('inf'):
                            continue
                        total_robustness += robustness
                    except Exception as e:
                        logger.error(f"Error in robustness calculation: {e}")
                        continue
                robustness_list.append(total_robustness)
            return robustness_list, indices, len(trajectory.cartesian.x)

        except Exception as e:
            logger.error(f"Error in robustness calculation: {e}, robustness_list: {robustness_list}")
            robustness_list = [0] * len(trajectory.cartesian.x)
            return robustness_list, indices

    # def _calculate_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
    #     """
    #     Compute the sum of rule robustness at specific points along the trajectory.
    #     """
    #     try:
    #         ego_vehicle = self.ego_vehicle
    #         robustness_list = []
    #         # Initialize RuleEvaluator with all the rules
    #         ego_vehicle_world_state = self.simulation.world_state.vehicle_by_id(-1)
    #         rule_evaluators = []
    #         for rule in self.robustness_types:
    #             rule_evaluator = RuleEvaluator.create_from_config(
    #                 self.simulation.world_state,
    #                 ego_vehicle_world_state,
    #                 rule
    #             )
    #             rule_evaluators.append(rule_evaluator)
    #
    #         num_points = len(trajectory.cartesian.x)
    #         indices = self._get_evaluation_indices(num_points)
    #         indices_set = set(indices)  # For faster lookup
    #         initial_time_step = copy.deepcopy(self.simulation.current_time_step)
    #
    #         for i in range(len(trajectory.cartesian.x) - 1):
    #             if i + 1 in indices_set:
    #                 # Transform coordinates
    #                 position_rear = np.array([trajectory.cartesian.x[i + 1], trajectory.cartesian.y[i + 1]])
    #                 orientation = trajectory.cartesian.theta[i + 1]
    #                 position_center = position_rear + ego_vehicle.parameters.b * np.array(
    #                     [math.cos(orientation), math.sin(orientation)])
    #                 time_step = initial_time_step + 1 + i
    #
    #                 state = InitialState(
    #                     position=position_center,
    #                     velocity=trajectory.cartesian.v[i + 1],
    #                     acceleration=trajectory.cartesian.a[i + 1],
    #                     orientation=orientation,
    #                     time_step=time_step,
    #                     yaw_rate=0.0,
    #                     slip_angle=0.0
    #                 )
    #
    #                 ego_vehicle.set_state(state)
    #                 self.simulation.jump_to_time_step(time_step)
    #
    #                 total_robustness = 0.0
    #                 for evaluator in rule_evaluators:
    #                     robustness_series = evaluator.evaluate()
    #                     robustness = robustness_series[-1]
    #                     # 如果robustness为inf，说明规则不适用，不计入总robustness
    #                     if robustness == float('inf'):
    #                         continue
    #                     total_robustness += robustness
    #                 robustness_list.append(total_robustness)
    #         return robustness_list, indices
    #
    #     except Exception as e:
    #         logger.error(f"Error in robustness calculation: {e}")
    #         return []

    def _get_evaluation_indices(self, num_points):
        """
        Get the indices corresponding to the specific points where robustness is evaluated.
        """
        idx_q1 = int(round(0.25 * (num_points - 1)))
        idx_mid = int(round(0.5 * (num_points - 1)))
        idx_q3 = int(round(0.75 * (num_points - 1)))
        idx_end = num_points - 1

        indices = [
            idx_q1,
            idx_mid,
            idx_q3,
            idx_end
        ]
        # Remove duplicates and sort
        indices = sorted(set(indices))
        return indices


def remove_future_states(ego_vehicle_world_state, initial_time_step):
    """
    检查并删除大于initial_time_step的状态
    """
    # 获取所有需要删除的时间步
    states_list = list(ego_vehicle_world_state.states_cr.values())
    states_to_remove = [
        state.time_step
        for state in states_list
        if state.time_step > initial_time_step
    ]

    # 删除每个大于initial_time_step的状态
    for time_step in states_to_remove:
        del ego_vehicle_world_state.states_cr[time_step]
        if time_step in ego_vehicle_world_state.lanelet_assignment:
            del ego_vehicle_world_state.lanelet_assignment[time_step]
        if time_step in ego_vehicle_world_state.signal_series:
            del ego_vehicle_world_state.signal_series[time_step]