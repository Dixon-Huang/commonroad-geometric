import matplotlib.animation as animation
import pickle
from typing import Iterable
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import LaneletNetwork
import os
from shapely import LineString
from projects.graph_rl_agents.predictive_traffic_rule_compliance.k_level.utility import distance_lanelet
from projects.graph_rl_agents.predictive_traffic_rule_compliance.k_level.detail_central_vertices import detail_cv
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3
import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.visualization.draw_params import MPDrawParams, StateParams, LaneletParams
from matplotlib.patches import Circle
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import InitialState, CustomState
import math
import random
import logging
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
import matplotlib.path as mpltPath
from shapely.geometry.point import Point
import copy
from projects.graph_rl_agents.predictive_traffic_rule_compliance.run_scenario_visualization import \
    visualize_scenario
import matplotlib.pyplot as plt
# from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, StateParams, LaneletParams
from matplotlib.patches import Circle


class Conf_Lanelet:
    '''冲突lanelet类a
        id: 路口内的冲突lanelet的id列表
        conf_point: 对应的冲突点位置 列表
    '''

    def __init__(self, id=None, conf_point=None):
        self.id = id
        self.conf_point = conf_point


def is_inside_lane(vehicle_pos, left_boundary, right_boundary):
    # 构建车道多边形，注意需要确保多边形顶点的顺序
    lane_polygon = np.vstack((left_boundary, right_boundary[::-1]))
    path = mpltPath.Path(lane_polygon)
    return path.contains_point(vehicle_pos)


# 定义动作枚举，添加新的 BRAKE 动作
class Action(Enum):
    MAINTAIN = [0.0, 0.0]  # 保持速度
    LOW_BRAKE = [-1.5, 0.0]  # 低速刹车
    LOW_ACCELERATE = [1.5, 0.0]  # 低速加速
    MID_BRAKE = [-3.5, 0.0]  # 中速刹车
    HIGH_ACCELERATE = [2.5, 0.0]  # 高速加速
    HIGH_BRAKE = [-5.0, 0.0]  # 高速刹车
    LOW_LEFT_STEER = [0.0, 0.25 * 3.14159]  # 小幅左转
    LOW_RIGHT_STEER = [0.0, -0.25 * 3.14159]  # 小幅右转
    HIGH_LEFT_STEER = [0.0, 0.5 * 3.14159]  # 大幅左转
    HIGH_RIGHT_STEER = [0.0, -0.5 * 3.14159]  # 大幅右转
    ACCELERATE_LEFT = [1.5, 0.25 * 3.14159]  # 加速 + 小幅左转
    ACCELERATE_RIGHT = [1.5, -0.25 * 3.14159]  # 加速 + 小幅右转
    BRAKE_LEFT = [-1.5, 0.25 * 3.14159]  # 刹车 + 小幅左转
    BRAKE_RIGHT = [-1.5, -0.25 * 3.14159]  # 刹车 + 小幅右转

    # MAINTAIN = [0.0, 0.0]
    # LEFT_STEER = [0.0, 0.25 * 3.14159]
    # RIGHT_STEER = [0.0, -0.25 * 3.14159]
    # ACCELERATE = [2.5, 0.0]
    # DECELERATE = [-2.5, 0.0]
    # BRAKE = [-5.0,0.0]


def has_overlap(box2d_0, box2d_1) -> bool:
    def get_edges(box):
        """获取多边形的所有边向量。"""
        x, y = box
        edges = []
        num_points = len(x)
        for i in range(num_points):
            dx = x[(i + 1) % num_points] - x[i]
            dy = y[(i + 1) % num_points] - y[i]
            edges.append((dx, dy))
        return edges

    def get_normal(edge):
        """获取边的法向量（分离轴）。"""
        dx, dy = edge
        return (-dy, dx)

    def project(box, axis):
        """将多边形投影到分离轴上，返回最小和最大投影值。"""
        projections = np.dot(axis, np.array(box))
        return projections.min(), projections.max()

    # 获取所有分离轴
    edges_0 = get_edges(box2d_0)
    edges_1 = get_edges(box2d_1)
    axes = [get_normal(edge) for edge in edges_0 + edges_1]

    for axis in axes:
        # 归一化分离轴（可选）
        axis = np.array(axis)
        axis_length = np.linalg.norm(axis)
        if axis_length == 0:
            continue  # 忽略零长度轴
        axis = axis / axis_length

        # 获取投影范围
        box0 = np.array(box2d_0)
        box1 = np.array(box2d_1)
        proj0_min, proj0_max = project(box0, axis)
        proj1_min, proj1_max = project(box1, axis)

        # 检查是否存在分离
        if proj0_max < proj1_min or proj1_max < proj0_min:
            # 存在分离轴，两个多边形不重叠
            return False

    # 没有找到分离轴，两个多边形重叠
    return True


# 更新动作列表，包含新的 BRAKE 动作
ActionList = [
    Action.MAINTAIN,
    Action.LOW_BRAKE,
    Action.LOW_ACCELERATE,
    Action.MID_BRAKE,
    Action.HIGH_ACCELERATE,
    Action.HIGH_BRAKE,
    Action.LOW_LEFT_STEER,
    Action.LOW_RIGHT_STEER,
    Action.HIGH_LEFT_STEER,
    Action.HIGH_RIGHT_STEER,
    Action.ACCELERATE_LEFT,
    Action.ACCELERATE_RIGHT,
    Action.BRAKE_LEFT,
    Action.BRAKE_RIGHT
]


# ActionList = [Action.MAINTAIN,Action.LEFT_STEER, Action.RIGHT_STEER,Action.ACCELERATE,Action.DECELERATE,Action.BRAKE]

class State:
    def __init__(self, x=0, y=0, yaw=0, v=0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def to_list(self) -> List:
        return [self.x, self.y, self.yaw, self.v]

    def copy(self):
        return State(self.x, self.y, self.yaw, self.v)


class StateList:
    def __init__(self, state_list=None) -> None:
        self.state_list: List[State] = state_list if state_list is not None else []

    def append(self, state: State) -> None:
        self.state_list.append(state)

    def reverse(self) -> 'StateList':
        self.state_list = self.state_list[::-1]
        return self

    def expand(self, excepted_len: int, expand_state: Optional[State] = None) -> None:
        cur_size = len(self.state_list)
        if cur_size >= excepted_len:
            return
        else:
            if expand_state is None:
                expand_state = self.state_list[-1]
            for _ in range(excepted_len - cur_size):
                self.state_list.append(expand_state)

    def to_list(self, is_vertical: bool = True) -> List:
        if is_vertical is True:
            states = [[], [], [], []]
            for state in self.state_list:
                states[0].append(state.x)
                states[1].append(state.y)
                states[2].append(state.yaw)
                states[3].append(state.v)
        else:
            states = []
            for state in self.state_list:
                states.append([state.x, state.y, state.yaw, state.v])
        return states

    def __len__(self) -> int:
        return len(self.state_list)

    def __getitem__(self, key: int) -> State:
        return self.state_list[key]

    def __setitem__(self, key: int, value: State) -> None:
        self.state_list[key] = value


class Node:
    MAX_LEVEL: int = 6
    calc_value_callback = None

    def __init__(self, state=State(), level=0, parent: Optional["Node"] = None,
                 action: Optional[Action] = None, others: List[State] = [],
                 goal: State = State()) -> None:
        self.state: State = state
        self.value: float = 0
        self.reward: float = 0
        self.visits: int = 0
        self.action: Action = action
        self.parent: Optional[Node] = parent
        self.cur_level: int = level
        self.goal_pos: State = goal

        self.children: List[Node] = []
        self.actions: List[Action] = []
        self.other_agent_state: List[State] = others
        self._is_terminal_flag: bool = False  # 新增内部标志

    @property
    def is_terminal(self) -> bool:
        return self.cur_level >= Node.MAX_LEVEL or self._is_terminal_flag

    @is_terminal.setter
    def is_terminal(self, value: bool):
        self._is_terminal_flag = value

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) >= len(ActionList)

    @staticmethod
    def initialize(max_level, callback) -> None:
        Node.MAX_LEVEL = max_level
        Node.calc_value_callback = callback

    def add_child(self, next_action: Action, delta_t: float, others: List[State] = []) -> "Node":
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, self, next_action, others, self.goal_pos)
        node.actions = self.actions + [next_action]
        Node.calc_value_callback(node, self.value)  # 传入 self.value 作为 last_node_value
        self.children.append(node)
        return node

    def next_node(self, delta_t: float, others: List[State] = []) -> "Node":
        next_action = random.choice(ActionList)
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, None, next_action, others, self.goal_pos)
        Node.calc_value_callback(node, self.value)
        return node

    def __repr__(self):
        return (f"children: {len(self.children)}, visits: {self.visits}, "
                f"reward: {self.reward}, actions: {self.actions}")


# 定义车辆基类
class VehicleBase:
    width = 1.8
    length = 4.3
    safe_width = 2.4
    safe_length = 4.9

    def __init__(self, vehicle_id, initial_state: State, goal_state: State, level: int, lane_lines=None):
        self.vehicle_id = vehicle_id
        self.state = initial_state
        self.goal_state = goal_state
        self.level = level
        self.lane_lines = lane_lines  # 添加车道线信息
        self.footprint = StateList([initial_state.copy()])  # 记录轨迹
        self.is_get_target = False

        self.width = 1.8
        self.length = 4.3

    @staticmethod
    def get_safezone(tar_offset: State) -> np.ndarray:
        safezone = np.array(
            [[-VehicleBase.safe_length / 2, VehicleBase.safe_length / 2,
              VehicleBase.safe_length / 2, -VehicleBase.safe_length / 2, -VehicleBase.safe_length / 2],
             [VehicleBase.safe_width / 2, VehicleBase.safe_width / 2,
              -VehicleBase.safe_width / 2, -VehicleBase.safe_width / 2, VehicleBase.safe_width / 2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                        [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        safezone = np.dot(rot, safezone)
        safezone += np.array([[tar_offset.x], [tar_offset.y]])

        return safezone

    @staticmethod
    def get_box2d(tar_offset: State) -> np.ndarray:
        vehicle = np.array(
            [[-VehicleBase.length / 2, VehicleBase.length / 2,
              VehicleBase.length / 2, -VehicleBase.length / 2, -VehicleBase.length / 2],
             [VehicleBase.width / 2, VehicleBase.width / 2,
              -VehicleBase.width / 2, -VehicleBase.width / 2, VehicleBase.width / 2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                        [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        vehicle = np.dot(rot, vehicle)
        vehicle += np.array([[tar_offset.x], [tar_offset.y]])

        return vehicle

    def update_state(self, action: Action, delta_t: float):
        self.state = kinematic_propagate(self.state, action.value, delta_t)
        self.footprint.append(self.state.copy())

        # 检查是否到达目标
        if self.reached_goal():
            self.is_get_target = True

    def copy_with_level(self, new_level: int):
        # 仅复制需要修改的属性
        new_initial_state = self.state.copy()  # 假设 State 类有 copy 方法
        new_goal_state = self.goal_state.copy()
        # 假设 lane_lines 是不可变的或不需要复制
        return VehicleBase(
            vehicle_id=self.vehicle_id,
            initial_state=new_initial_state,
            goal_state=new_goal_state,
            level=new_level,
            lane_lines=self.lane_lines  # 如果需要修改，可以进行深拷贝
        )

    def reached_goal(self) -> bool:
        position_error = np.hypot(self.state.x - self.goal_state.x, self.state.y - self.goal_state.y)
        orientation_error = abs(self.state.yaw - self.goal_state.yaw)
        if position_error < 1.0 and orientation_error < np.deg2rad(10):
            return True
        else:
            return False


def normalize_angle(angle):
    """
    将角度归一化到 [-π, π] 范围内。

    参数:
        angle (float): 要归一化的角度，单位为弧度。

    返回:
        float: 归一化后的角度。
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# 定义蒙特卡洛树搜索类
class MonteCarloTreeSearch:
    EXPLORATE_RATE = 1 / (2 * math.sqrt(2.0))

    def __init__(self, ego: VehicleBase, others: List[VehicleBase],
                 other_traj: List[StateList], cfg: dict = {}):
        self.ego_vehicle: VehicleBase = ego
        self.other_vehicle: List[VehicleBase] = others
        self.other_predict_traj: List[StateList] = other_traj
        self.computation_budget = cfg.get('computation_budget', 1000)
        self.dt = cfg.get('delta_t', 0.25)

        self.LAMBDA = cfg.get('lambda', 0.9)
        self.WEIGHT_AVOID = cfg.get('weight_avoid', 10)
        self.WEIGHT_SAFE = cfg.get('weight_safe', 0.2)
        self.WEIGHT_OFFROAD = cfg.get('weight_offroad', 2)
        self.WEIGHT_DIRECTION = cfg.get('weight_direction', 1)
        self.WEIGHT_DISTANCE = cfg.get('weight_distance', 0.1)
        self.WEIGHT_VELOCITY = cfg.get('weight_velocity', 0.05)
        self.WEIGHT_REVERSE = cfg.get('weight_reverse', 5)
        self.WEIGHT_GOAL_DISTANCE = cfg.get('weight_goal_distance', 1.0)
        self.WEIGHT_TIME = cfg.get('weight_time', 0.5)

    def excute(self, root: Node) -> Node:
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward
            self.update(expand_node, reward)
        return self.get_best_child(root, 0)

    def tree_policy(self, node: Node) -> Node:
        while not node.is_terminal:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < 0.5:
                node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)
            else:
                if not node.is_fully_expanded:
                    return self.expand(node)
                else:
                    node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)
        return node

    def default_policy(self, node: Node) -> float:
        current_node = node
        total_reward = 0.0
        discount = 1.0
        last_node_value = node.value
        while not current_node.is_terminal and current_node.cur_level < Node.MAX_LEVEL:
            # Randomly choose an action for ego vehicle
            next_action = random.choice(ActionList)
            # Get predicted states of other vehicles
            next_level = current_node.cur_level + 1
            if next_level >= len(self.other_predict_traj):
                logging.warning(f"超出预测轨迹范围，使用最后一个预测状态。")
                cur_other_state = self.other_predict_traj[-1]
            else:
                cur_other_state = self.other_predict_traj[next_level]
            # Update ego vehicle state
            new_state = kinematic_propagate(current_node.state, next_action.value, self.dt)
            # Create new node with updated state and other vehicles' states
            current_node = Node(new_state, parent=current_node, action=next_action, level=next_level + 1,
                                others=cur_other_state, goal=current_node.goal_pos)
            # Calculate reward considering other vehicles
            self.calc_cur_value(current_node, last_node_value)
            last_node_value = current_node.value
            # Termination condition
            if self.check_termination(current_node.state):
                current_node.is_terminal = True
        return current_node.value

    def update(self, node: Node, r: float) -> None:
        while node is not None:
            node.visits += 1
            node.reward += r
            node = node.parent

    def expand(self, node: Node) -> Node:
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in ActionList if a not in tried_actions]
        if not untried_actions:
            #node.is_fully_expanded = True
            return node  # 如果没有未尝试的动作，返回当前节点
        next_action = random.choice(untried_actions)
        # 获取下一个他车状态
        if node.cur_level + 1 >= len(self.other_predict_traj):
            logging.warning(f"超出预测轨迹范围，使用最后一个预测状态。")
            cur_other_state = self.other_predict_traj[-1]
        else:
            cur_other_state = self.other_predict_traj[node.cur_level + 1]
        # 添加子节点
        child_node = node.add_child(next_action, self.dt, cur_other_state)
        return child_node  # 返回新添加的子节点

    def get_best_child(self, node: Node, scalar: float) -> Node:
        best_score = -math.inf
        best_children = []
        for child in node.children:
            if child.visits == 0:
                continue
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            score = exploit + scalar * explore
            if score > best_score:
                best_children = [child]
                best_score = score
            elif score == best_score:
                best_children.append(child)
        if not best_children:
            return node
        return random.choice(best_children)

    def calc_cur_value(self, node: Node, last_node_value: float) -> float:
        # 获取车辆的车道线信息
        lane_lines = self.ego_vehicle.lane_lines
        if lane_lines is None:
            # 如果没有车道线信息，无法计算奖励，返回一个大的惩罚
            node.value = -1000
            return node.value

        # 车辆的位置和航向
        x, y, yaw = node.state.x, node.state.y, node.state.yaw

        # 转换车道线信息为 NumPy 数组
        center_line = np.array(lane_lines['center_line'])
        left_boundary = np.array(lane_lines['left_bound'])
        right_boundary = np.array(lane_lines['right_bound'])

        vehicle_position = np.array([x, y])

        # 计算车辆与中心线的距离
        distances = np.linalg.norm(center_line - vehicle_position, axis=1)
        idx = np.argmin(distances)
        closest_point = center_line[idx]
        distance_to_center = distances[idx]

        # 计算车道中心线在最近点的切线方向
        num_points = len(center_line)
        if num_points < 2:
            # 无法计算切线方向，使用默认方向
            tangent_vector = np.array([1.0, 0.0])
        else:
            if idx == 0:
                tangent_vector = center_line[1] - center_line[0]
            elif idx == num_points - 1:
                tangent_vector = center_line[-1] - center_line[-2]
            else:
                tangent_vector = center_line[idx + 1] - center_line[idx - 1]

        # 计算切线方向的航向角
        tangent_yaw = np.arctan2(tangent_vector[1], tangent_vector[0])

        # 计算车辆航向与车道切线方向的偏差
        # 计算车辆航向与车道切线方向的偏差
        yaw_normalized = normalize_angle(yaw)
        tangent_yaw_normalized = normalize_angle(tangent_yaw)
        delta_yaw = normalize_angle(yaw_normalized - tangent_yaw_normalized)
        delta_yaw_abs = abs(delta_yaw)
        #direction_reward = -delta_yaw_abs * self.WEIGHT_DIRECTION
        # 航向偏差奖励使用指数衰减
        direction_reward = -np.exp(delta_yaw_abs) * self.WEIGHT_DIRECTION
        # 距离中心线的奖励（距离越小，奖励越高）
        distance_reward = -distance_to_center * self.WEIGHT_DISTANCE

        # 使用多边形的方法，判断点是否在多边形内
        # 构建车道多边形
        lane_polygon = np.vstack((left_boundary, right_boundary[::-1]))
        lane_path = mpltPath.Path(lane_polygon)

        is_inside_lane = lane_path.contains_point(vehicle_position)

        # 超出车道，施加惩罚
        offroad_penalty = -self.WEIGHT_OFFROAD if not is_inside_lane else 0.0

        # 检查车辆速度是否为负（反向行驶）
        reverse_penalty = -self.WEIGHT_REVERSE if node.state.v < -0.1 else 0.0

        final_speed = self.ego_vehicle.goal_state.v  # 假设您有末速度
        velocity_error = node.state.v / final_speed
        velocity_reward = velocity_error * self.WEIGHT_VELOCITY

        # 避免与其他车辆碰撞
        avoid = 0
        safe = 0
        ego_box2d = VehicleBase.get_box2d(node.state)
        ego_safezone = VehicleBase.get_safezone(node.state)

        for cur_other_state in node.other_agent_state:
            other_box2d = VehicleBase.get_box2d(cur_other_state)
            other_safezone = VehicleBase.get_safezone(cur_other_state)
            if has_overlap(ego_box2d, other_box2d):
                avoid -= 1
            if has_overlap(ego_safezone, other_safezone):
                safe -= 1

        avoid_reward = self.WEIGHT_AVOID * avoid
        safe_reward = self.WEIGHT_SAFE * safe

        # 到目标的距离奖励
        position_error = np.hypot(x - self.ego_vehicle.goal_state.x, y - self.ego_vehicle.goal_state.y)
        goal_distance_reward = -position_error * self.WEIGHT_GOAL_DISTANCE

        # 时间惩罚
        time_penalty = -node.cur_level * self.WEIGHT_TIME

        # 总奖励
        if self.check_termination(node.state):
            cur_reward = 1000.0  # 到达目标的高额奖励
        else:
            cur_reward = (distance_reward +
                          offroad_penalty +
                          reverse_penalty +
                          direction_reward +
                          velocity_reward +
                          avoid_reward +
                          safe_reward +
                          goal_distance_reward +
                          time_penalty
                          )
        #print(distance_reward,offroad_penalty,reverse_penalty,direction_reward,velocity_reward,avoid_reward,safe_reward,goal_distance_reward,time_penalty)
        # 更新节点的即时奖励
        #node.value = cur_reward
        step = node.cur_level
        total_reward = last_node_value + (self.LAMBDA ** (step - 1)) * cur_reward
        node.value = total_reward
        return total_reward
        #return cur_reward

    def check_termination(self, state: State) -> bool:
        # 判断是否到达目标
        position_error = np.hypot(state.x - self.ego_vehicle.goal_state.x, state.y - self.ego_vehicle.goal_state.y)
        if position_error < 1.0:
            return True
        # 可以添加碰撞检测逻辑
        return False


# 定义 Level-k 规划器
class KLevelPlanner:
    def __init__(self, cfg: dict = {}):
        self.steps = cfg.get('max_step', 8)
        self.dt = cfg.get('delta_t', 0.25)
        self.config = cfg

    def planning(self, ego: VehicleBase, others: List[VehicleBase]) -> Tuple[Action, StateList]:
        # 为当前车辆创建 MCTS 对象
        other_prediction = self.get_prediction(ego, others)
        mcts = MonteCarloTreeSearch(ego, others, other_prediction, self.config)
        root = Node(state=ego.state.copy(), goal=ego.goal_state)
        Node.initialize(self.steps, mcts.calc_cur_value)
        best_node = mcts.excute(root)
        # 回溯最佳路径
        actions = []
        current_node = best_node
        state_list = StateList()
        while current_node.parent is not None:
            actions.append(current_node.action)
            state_list.append(current_node.state)
            current_node = current_node.parent
        actions.reverse()
        state_list.reverse()

        if len(actions) < self.steps:
            # 如果动作序列不足，使用保持动作填充
            actions.extend([Action.MAINTAIN] * (self.steps - len(actions)))
            state_list.expand(self.steps + 1)

        return actions, state_list

    def forward_simulate(self, ego: VehicleBase, others: List[VehicleBase],
                         traj: List[StateList]) -> Tuple[List[Action], StateList]:
        mcts = MonteCarloTreeSearch(ego, others, traj, self.config)
        root = Node(state=ego.state.copy(), goal=ego.goal_state)
        Node.initialize(self.steps, mcts.calc_cur_value)
        best_node = mcts.excute(root)
        current_node = best_node
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0)
        # 回溯最佳路径
        actions = []
        current_node = best_node
        state_list = StateList()
        while current_node.parent is not None:
            actions.append(current_node.action)
            state_list.append(current_node.state)
            current_node = current_node.parent
        actions.reverse()
        state_list.reverse()

        if len(actions) < self.steps:
            # 如果动作序列不足，使用保持动作填充
            actions.extend([Action.MAINTAIN] * (self.steps - len(actions)))
            state_list.expand(self.steps + 1)

        return actions, state_list

    def get_prediction(self, ego: VehicleBase, others: List[VehicleBase], depth=0) -> List[StateList]:
        indent = "  " * depth  # 根据递归深度添加缩进
        pred_trajectory = []  # 最终预测轨迹
        pred_trajectory_trans = []  # 交换层级后的轨迹

        # Level-0 车辆：直接复制其他车辆状态
        if ego.level == 0:
            for step in range(self.steps + 1):
                pred_traj = StateList([other.state for other in others])
                pred_trajectory.append(pred_traj)
            return pred_trajectory

        # Level-k 推理处理逻辑 (k > 0)
        for idx, other in enumerate(others):
            if other.is_get_target:
                # 如果车辆是目标车辆，直接复制其状态
                pred_traj = StateList([other.state] * (self.steps + 1))
                pred_trajectory_trans.append(pred_traj)
                continue

            # 将当前其他车辆与 ego 交换层级，生成新的车辆副本
            exchanged_ego = other.copy_with_level(max(ego.level - 1, 0))
            exchanged_others = [ego] + [v for i, v in enumerate(others) if i != idx]

            # 递归获取交换后车辆的预测轨迹
            exchanged_pred = self.get_prediction(exchanged_ego, exchanged_others, depth + 1)

            # 前向模拟获取预测的车辆状态
            _, pred_vehicle = self.forward_simulate(exchanged_ego, exchanged_others, exchanged_pred)
            pred_trajectory_trans.append(pred_vehicle)

        # 合并所有预测轨迹
        for i in range(self.steps + 1):
            state = StateList([states[i] for states in pred_trajectory_trans])
            pred_trajectory.append(state)

        return pred_trajectory


# 定义运动学传播函数
def kinematic_propagate(state: State, act: List[float], dt: float) -> State:
    next_state = State()
    acc, omega = act[0], act[1]
    next_state.x = state.x + state.v * np.cos(state.yaw) * dt
    next_state.y = state.y + state.v * np.sin(state.yaw) * dt
    next_state.v = state.v + acc * dt
    next_state.yaw = state.yaw + omega * dt

    #next_state.yaw = (next_state.yaw + 2 * np.pi) % (2 * np.pi)
    next_state.v = max(min(next_state.v, 20), 0)
    return next_state


def compute_vehicle_level(vehicle_state: State, other_vehicle_state: State, ego_vehicle: VehicleBase,
                          collision_point: Point) -> int:
    ego_state = ego_vehicle.state

    def distance_to_vehicle(state1: State, state2: State) -> float:
        """计算两个车辆之间的距离"""
        return np.hypot(state1.x - state2.x, state1.y - state2.y)

    def distance_to_collision(state: State, collision_point: Point) -> float:
        """计算车辆到碰撞点的距离"""
        return np.hypot(collision_point.x - state.x, collision_point.y - state.y)

    def is_approaching(state: State, collision_point: Point) -> bool:
        """判断车辆是否朝向碰撞点移动"""
        to_collision = np.array([collision_point.x - state.x, collision_point.y - state.y])
        distance = np.linalg.norm(to_collision)

        if distance == 0:
            return False  # 车辆已在碰撞点，不再接近

        velocity = np.array([state.v * np.cos(state.yaw), state.v * np.sin(state.yaw)])
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm == 0:
            return False  # 车辆静止，不再接近

        to_collision_unit = to_collision / distance
        velocity_unit = velocity / velocity_norm
        dot_product = np.dot(to_collision_unit, velocity_unit)
        return dot_product > 0  # 如果点积大于0，表示车辆朝向碰撞点移动

    def has_passed_collision_point(state: State, collision_point: Point) -> bool:
        """判断车辆是否已经通过了碰撞点"""
        to_collision = np.array([collision_point.x - state.x, collision_point.y - state.y])
        distance = np.linalg.norm(to_collision)

        if distance == 0:
            return False  # 车辆在冲撞点，未通过

        velocity = np.array([state.v * np.cos(state.yaw), state.v * np.sin(state.yaw)])
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm == 0:
            return False  # 车辆静止，无法判断

        to_collision_unit = to_collision / distance
        velocity_unit = velocity / velocity_norm
        dot_product = np.dot(to_collision_unit, velocity_unit)
        return dot_product < 0  # 如果点积小于0，表示车辆已通过碰撞点

    # 计算车辆到冲撞点的距离和与自车的距离
    distance_to_collision_point = distance_to_collision(vehicle_state, collision_point)
    distance_v2v = distance_to_vehicle(vehicle_state, ego_state)
    approaching = is_approaching(vehicle_state, collision_point)
    passed_collision_point = has_passed_collision_point(vehicle_state, collision_point)

    # 条件1：车辆正在向冲撞点移动，并且距离冲撞点小于或等于10米
    condition1 = approaching and distance_to_collision_point <= 10

    # 条件2：车辆与自车的距离小于或等于5米
    condition2 = distance_v2v <= 5

    if condition1 or condition2:
        # 根据速度判断level
        speeds = [ego_state.v, vehicle_state.v, other_vehicle_state.v]
        sorted_speeds = sorted(speeds, reverse=True)

        if vehicle_state.v >= sorted_speeds[0]:
            # 速度在三车中最快，设置level为0
            print('d2: Fastest speed among the three')
            return 0
        elif vehicle_state.v >= ego_state.v:
            # 速度比ego快，Level减一，最小为0
            new_level = max(ego_vehicle.level - 1, 0)
            print(f'd3: Speed >= ego speed, setting level to {new_level}')
            return new_level
        else:
            # 速度比ego慢，Level加一，最大为2
            new_level = min(ego_vehicle.level + 1, 2)
            print(f'd4: Speed < ego speed, setting level to {new_level}')
            return new_level

    # 如果车辆正在向冲撞点移动，并且距离冲撞点大于10米，或者已通过冲撞点且与自车距离大于5米，则设置level为0
    if (approaching and distance_to_collision_point > 10) or (passed_collision_point and distance_v2v > 5):
        print('d1: Passed collision point and distance >= 5 meters or approaching but distance >10 meters')
        return 0

    # 其他情况，保持当前level
    print('d5: Not within 10 meters or not approaching collision point')
    return 0


def sort_conf_point(ego_pos, dict_lanelet_conf_point, cv, cv_s):
    """ 给冲突点按照离自车的距离 由近到远 排序。
    params:
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    returns:
        sorted_lanelet: conf_points 的下标排序
        i_ego: sorted_lanelet[i_ego]则是自车需要考虑的最近的lanelet
    """
    conf_points = list(dict_lanelet_conf_point.values())
    lanelet_ids = np.array(list(dict_lanelet_conf_point.keys()))
    distance = []
    for conf_point in conf_points:
        distance.append(distance_lanelet(cv, cv_s, ego_pos, conf_point))
    distance = np.array(distance)
    id = np.argsort(distance)

    distance_sorted = distance[id]
    index = np.where(distance_sorted > 0)[0]
    if len(index) == 0:
        i_ego = len(distance)
    else:
        i_ego = index.min()

    sorted_lanelet = lanelet_ids[id]

    return sorted_lanelet, i_ego


def conf_lanelet_checker(ln, sub_lanelet_id: int, lanelet_state: int, lanelet_route):
    # change incoming id into current lanelet id and add
    # if ego car is in incoming or in intersection
    """
    check conflict lanelets when driving in a incoming lanelet

    :param ln: lanelet network of the scenario
    :param sub_lanelet_id: the lanelet that subjective vehicle is located
    :param lanelet_state: 1:straight-going /2:incoming /3:in-intersection/4:straight-going in intersection
    :param lanelet_route: lanelet route
    :return: Conf_lanelet类【.id:路口内的冲突lanelet的id；.conf_point:对应的冲突点位置】
    """

    def check_sub_car_in_incoming():
        """ 返回主车所在的intersection序号和incoming序号 """
        route = lanelet_route
        # 初始化
        idx_intersect = []  # 主车所在的路口的序号
        lanelet_id_in_intersection = []  # 主车即将经过的路口中的lanelet的id
        sub_lanelet_id_incoming = []

        if lanelet_state == 2:
            id_temp = route.lanelet_ids.index(sub_lanelet_id)
            # id_temp = ln.find_lanelet_by_id(sub_lanelet_id)
            if id_temp + 1 < len(route.lanelet_ids):  # incoming lanelet is NOT the last one in route
                lanelet_id_in_intersection = route.lanelet_ids[id_temp + 1]

            sub_lanelet_id_incoming = sub_lanelet_id  # 找到主车进入路口的incoming,用于确定路口序号

        elif lanelet_state == 3:
            lanelet_id_in_intersection = sub_lanelet_id  # 此时主车已经在路口内，不用找id_in_intersection了
            print(ln.find_lanelet_by_id(sub_lanelet_id).predecessor)
            if ln.find_lanelet_by_id(sub_lanelet_id).predecessor:
                print('ttttttttttt', ln.find_lanelet_by_id(sub_lanelet_id).predecessor)
                sub_lanelet_id_incoming = ln.find_lanelet_by_id(sub_lanelet_id).predecessor[
                    0]  # 找到主车进入路口的incoming,用于确定路口序号

        # 遍历场景中的路口，找到主车所在的路口
        for idx, intersection in enumerate(ln.intersections):
            # 遍历路口中的每个incoming
            incomings = intersection.incomings
            for n, incoming in enumerate(incomings):
                lanelet_id_list = setToArray(incoming.incoming_lanelets)
                # if np.isin(sub_lanelet_id_incoming, lanelet_id_list):  # 检查主车所在lanelet对应的incoming id

                # Change
                # if np.isin(sub_lanelet_id, lanelet_id_list):
                if np.isin(sub_lanelet_id_incoming, lanelet_id_list):
                    # 记录主车所在的【路口序号】
                    idx_intersect = idx

        return idx_intersect, lanelet_id_in_intersection

    def check_in_intersection_lanelets(lanelet_id_in_intersection):
        """ 返回提取路口内所有的lanelet的id列表 """

        lanelet_list = list()

        ego_incoming = ln.find_lanelet_by_id(lanelet_id_in_intersection).predecessor

        interscetionlist = list(lanelet_network.intersections)
        if interscetionlist != [] and id_intersect >= 0:
            intersection = interscetionlist[id_intersect]  # 主车所在路口
            incomings = intersection.incomings

            for n in range(len(incomings)):

                incoming = incomings[n]
                if (list(incoming.incoming_lanelets) == ego_incoming): continue
                if len(incoming.successors_left):
                    # print("incoming_id =", n, "left successor exists")
                    list_temp = list(incoming.successors_left)
                    lanelet_list.append(list_temp[0])
                if len(incoming.successors_straight):
                    # print("incoming_id =", n, "straight successor exists")
                    list_temp = list(incoming.successors_straight)
                    lanelet_list.append(list_temp[0])
                if len(incoming.successors_right):
                    # print("incoming_id =", n, "right successor exists")
                    list_temp = list(incoming.successors_right)
                    lanelet_list.append(list_temp[0])
        return lanelet_list

    def check_collision(cv_sub_origin, cv_other_origin):
        """检测两条中线之间是否存在冲突，返回冲突是否存在，以及冲突点位置（若存在） """
        # faster version
        cv_sub_str = LineString(cv_sub_origin)
        cv_other_str = LineString(cv_other_origin)
        conf_point = cv_sub_str.intersection(cv_other_str)
        if conf_point:
            isconf = 1
            # print(conf_point)
        else:
            isconf = 0
        return isconf, conf_point

    def check_conf_lanelets(lanelet_network, laneletid_list: list, sub_lanelet_id_in_intersection: int):
        """ 返回存在冲突的lanelet列表（id,冲突位置） """
        # 创建一个冲突lanelet的类，记录冲突信息
        cl = Conf_Lanelet()

        if sub_lanelet_id_in_intersection:
            lanelet_sub = lanelet_network.find_lanelet_by_id(sub_lanelet_id_in_intersection)  # 主车的lanelet

            # 列出其他lanelet
            other_lanelet = list()
            for n in range(len(laneletid_list)):
                lanelet = lanelet_network.find_lanelet_by_id(laneletid_list[n])
                if not lanelet.lanelet_id == sub_lanelet_id_in_intersection:
                    other_lanelet.append(lanelet)

            cl.id = []
            cl.conf_point = []
            for n in range(len(other_lanelet)):
                [isconf, conf_point] = check_collision(lanelet_sub.center_vertices, other_lanelet[n].center_vertices)
                if isconf:
                    cl.id.append(other_lanelet[n].lanelet_id)
                    cl.conf_point.append(conf_point)
        else:

            cl.id = []
            cl.conf_point = []

        return cl

    # 主程序
    # 检查主车所在的路口和incoming序号
    [id_intersect, sub_lanelet_id_in_intersection] = check_sub_car_in_incoming()
    # print("intersection no.", id_intersect)
    # print("incoming_lanelet_id", id_incoming)
    # print("lanelet id of subjective car:", sub_lanelet_id_in_intersection)

    lanelet_network = ln
    # 提取路口内的lanelet的id列表
    inter_laneletid_list = check_in_intersection_lanelets(sub_lanelet_id_in_intersection)
    # print("all lanelets:", inter_laneletid_list)

    # 检查主车在路口内所需通过的lanelet与其他路口内的lanelet的冲突情况
    cl = check_conf_lanelets(lanelet_network, inter_laneletid_list, sub_lanelet_id_in_intersection)
    print("conflict lanelets", cl.id)
    return cl


def potential_conf_lanelet_checkerv2(lanelet_network, cl_info):
    """
    check [potential] conflict lanelets when driving in a incoming lanelet

    :param ln: lanelet network of the scenario
    :param cl_info: 直接冲突lanelet的Conf_lanelet类【.id:路口内的冲突lanelet的id；.conf_point:对应的冲突点位置】
    :return: dict_lanelet_parent. 间接冲突lanelet->直接冲突lanelet*列表*。(间接冲突是直接冲突的parent，一个间接可能对应多个直接)
    """

    dict_parent_lanelet = {}
    for conf_lanelet_id in cl_info.id:
        conf_lanlet = lanelet_network.find_lanelet_by_id(conf_lanelet_id)
        parents = conf_lanlet.predecessor
        if parents is not None:
            # 对于所有直接冲突lanelet的父节点。寻找他所有可能的子节点
            for parent in parents:
                if parent not in dict_parent_lanelet.keys():
                    parent_lanelet = lanelet_network.find_lanelet_by_id(parent)

                    child_lanelet_ids = parent_lanelet.successor
                    dict_parent_lanelet[parent] = child_lanelet_ids

    return dict_parent_lanelet


def conf_agent_checker(lanelet_network, dict_lanelet_conf_points, scenario):
    """  找直接冲突点 conf_lanelets中最靠近冲突点的车，为冲突车辆
    params:
        dict_lanelet_conf_points: 字典。直接冲突点的lanelet_id->冲突点位置
        T: 仿真时间步长
    returns:
        [!!!若该lanelet上没有障碍物，则没有这个lanelet的key。]
        字典dict_lanelet_agent: lanelet-> obstacle_id。可以通过scenario.obstacle_by_id(obstacle_id)获得该障碍物。
        [option] 非必要字典dict_lanelet_d: lanelet - > distance。障碍物到达冲突点的距离。负数说明过了冲突点一定距离
    """

    conf_lanelet_ids = list(dict_lanelet_conf_points.keys())  # 所有冲突lanelet列表

    dict_lanelet_agent = {}  # 字典。key: lanelet, obs_id ;
    dict_lanelet_d = {}  # 字典。key: lanelet, value: distacne .到冲突点的路程

    n_obs = len(scenario.obstacles)
    # 暴力排查场景中的所有车
    for i in range(n_obs):
        state = scenario.obstacles[i].state_at_time(0)  # zxc:scenario是实时的，所有T都改成0
        # 当前时刻这辆车可能没有
        if state is None:
            continue
        pos = scenario.obstacles[i].state_at_time(0).position
        lanelet_ids = lanelet_network.find_lanelet_by_position([pos])[0]
        # 可能在多条车道上，现在每个都做检查
        for lanelet_id in lanelet_ids:
            # 不能仅用位置判断车道。车的朝向也需要考虑?暂不考虑朝向。因为这样写不美。可能在十字路口倒车等
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            # 用自带的函数，检查他车是否在该lanelet上
            res = lanelet.get_obstacles([scenario.obstacles[i]], 0)
            if scenario.obstacles[i] not in res:
                continue

            # 如果该车在 冲突lanelet上
            if lanelet_id in conf_lanelet_ids:
                lanelet_center_line = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

                # 插值函数
                lanelet_center_line, _, lanelet_center_line_s = detail_cv(lanelet_center_line)

                conf_point = dict_lanelet_conf_points[lanelet_id]
                conf_point = np.array([conf_point.x, conf_point.y])
                d_obs2conf_point = distance_lanelet(lanelet_center_line, lanelet_center_line_s, pos, conf_point)

                # 车辆已经通过冲突点，跳过循环
                # 可能有问题...在冲突点过了一点点的车怎么搞？
                if d_obs2conf_point < -2 - scenario.obstacles[i].obstacle_shape.length / 2:
                    # 如果超过冲突点一定距离。不考虑该车
                    break
                if lanelet_id not in dict_lanelet_d:
                    # 该lanelet上出现的第一辆车
                    dict_lanelet_d[lanelet_id] = d_obs2conf_point
                    dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id
                else:
                    if d_obs2conf_point < dict_lanelet_d[lanelet_id]:
                        dict_lanelet_d[lanelet_id] = d_obs2conf_point
                        dict_lanelet_agent[lanelet_id] = scenario.obstacles[i].obstacle_id

    return dict_lanelet_agent


def potential_conf_agent_checker(lanelet_network, dict_lanelet_conf_point, dict_parent_lanelet, ego_lanelets, scenario):
    '''找间接冲突lanelet.
    params:
        dict_lanelet_conf_point: intersectioninfo类成员。
        dict_parent_lanelet: 间接冲突lanelet->子节点列表。
        T:
    returns:
        dict_lanelet_potential_agent: 间接冲突lanelet->冲突智能体列表。
    '''
    # 即使一辆车多个意图可能相撞，但是只用取一个值就行。任一个冲突点都是靠近终点。

    dict_parent_conf_point = {}  # 可能冲突lanelet -> 随意一个冲突点；因为越靠近终点的就是最需要的车辆。
    for parent, kids in dict_parent_lanelet.items():
        for kid in kids:
            if kid in dict_lanelet_conf_point.keys():
                dict_parent_conf_point[parent] = dict_lanelet_conf_point[kid]

    # 删除前车影响：
    for ego_lanelet in ego_lanelets.lanelet_ids:
        if ego_lanelet in dict_parent_conf_point.keys():
            dict_parent_conf_point.pop(ego_lanelet)

    dict_lanelet_potential_agent = conf_agent_checker(lanelet_network, dict_parent_conf_point, scenario)

    return dict_lanelet_potential_agent


class Ipaction():
    def __init__(self):
        self.v_end = 0
        self.a_end = 0
        self.delta_s = None
        self.frenet_cv = []
        self.T = None
        self.ego_state_init = []
        self.lanelet_id_target = None


def get_route_frenet_line(route, lanelet_network):
    ''' 获取route lanelt id的对应参考线
    '''
    # TODO: 是否存在route是多个左右相邻的并排车道的情况。此时不能将lanelet的中心线直接拼接
    cv = []
    for n in range(len(route)):
        if n == 0:
            cv = lanelet_network.find_lanelet_by_id(route[n]).center_vertices
        else:
            cv_temp = lanelet_network.find_lanelet_by_id(route[n]).center_vertices
            cv = np.concatenate((cv, cv_temp), axis=0)
    ref_cv, ref_orientation, ref_s = detail_cv(cv)
    ref_cv = np.array(ref_cv).T
    return ref_cv, ref_orientation, ref_s


def sort_conf_point(ego_pos, dict_lanelet_conf_point, cv, cv_s):
    """ 给冲突点按照离自车的距离 由近到远 排序。
    params:
        center_line: 道路中心线；
        s : 道路中心线累积距离;
        p1, p2: 点1， 点2
    returns:
        sorted_lanelet: conf_points 的下标排序
        i_ego: sorted_lanelet[i_ego]则是自车需要考虑的最近的lanelet
    """
    conf_points = list(dict_lanelet_conf_point.values())
    lanelet_ids = np.array(list(dict_lanelet_conf_point.keys()))
    distance = []
    for conf_point in conf_points:
        distance.append(distance_lanelet(cv, cv_s, ego_pos, conf_point))
    distance = np.array(distance)
    id = np.argsort(distance)

    distance_sorted = distance[id]
    index = np.where(distance_sorted > 0)[0]
    if len(index) == 0:
        i_ego = len(distance)
    else:
        i_ego = index.min()

    sorted_lanelet = lanelet_ids[id]

    return sorted_lanelet, i_ego


def find_reference(s, ref_cv, ref_orientation, ref_cv_len):
    ref_cv, ref_orientation, ref_cv_len = np.array(ref_cv), np.array(ref_orientation), np.array(ref_cv_len)
    id = np.searchsorted(ref_cv_len, s)
    if id >= ref_orientation.shape[0]:
        # print('end of reference line, please stop !')
        id = ref_orientation.shape[0] - 1
    return ref_cv[id, :], ref_orientation[id]


def front_vehicle_info_extraction(scenario, ego_pos, lanelet_route):
    '''lanelet_route第一个是自车车道。route直接往前找，直到找到前车。
    新思路：利用函数`find_lanelet_successors_in_range`寻找后继的lanelet节点。寻找在这些节点
    return:
        front_vehicle: dict. key: pos, vel, distance
    '''
    ln = scenario.lanelet_network
    front_vehicle = {}
    ref_cv, ref_orientation, ref_s = get_route_frenet_line(lanelet_route, ln)
    min_dhw = 500
    s_ego = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], ego_pos)
    lanelet_ids_ego = ln.find_lanelet_by_position([ego_pos])[0]
    # assert lanelet_ids_ego[0] in lanelet_route
    obstacles = scenario.obstacles
    for obs in obstacles:
        if obs.state_at_time(0):
            pos = obs.state_at_time(0).position
            # print(ln.find_lanelet_by_position([pos]))
            if not ln.find_lanelet_by_position([pos]) == [[]]:
                obs_lanelet_id = ln.find_lanelet_by_position([pos])[0][0]
                if obs_lanelet_id not in lanelet_route:
                    continue
            s_obs = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], pos)
            dhw = s_obs - s_ego
            if dhw < 0:
                continue
            if dhw < min_dhw:
                min_dhw = dhw
                front_vehicle['id'] = obs.obstacle_id
                front_vehicle['dhw'] = dhw
                front_vehicle['v'] = obs.state_at_time(0).velocity
                front_vehicle['state'] = obs.state_at_time(0)

    if len(front_vehicle) == 0:
        print('no front vehicle')
        front_vehicle['dhw'] = -1
        front_vehicle['v'] = -1

    return front_vehicle


def setToArray(setInput):
    arrayOutput = np.zeros((len(setInput), 1))
    index = 0
    for every in setInput:
        arrayOutput[index][0] = every
        index += 1
    return arrayOutput


class IntersectionInfo():
    ''' 提取交叉路口的冲突信息
    '''

    def __init__(self, cl) -> None:
        '''
        params:
            cl: Conf_Lanelet类
        '''
        self.dict_lanelet_conf_point = {}  # 直接冲突lanelet(与自车轨迹存在直接相交的lanelet,必定在路口内) ->冲突点坐标
        for i in range(len(cl.id)):
            self.dict_lanelet_conf_point[cl.id[i]] = cl.conf_point[i]
        self.dict_agent_goal_points = {}
        self.dict_lanelet_agent = {}  # 场景信息。直接冲突lanelet - > 离冲突点最近的agent
        self.dict_parent_lanelet = {}  # 地图信息。间接冲突lanelet->直接冲突lanelet*列表*。(间接冲突是直接冲突的parent，一个间接可能对应多个直接)
        self.dict_lanelet_potential_agent = {}  # 间接冲突lanelet - > 离冲突点最近的agent。
        self.sorted_lanelet = []  # 直接冲突lanelet按照冲突点位置进行排序。
        self.i_ego = 0  # 自车目前通过了哪个冲突点。0代表在第一个冲突点之前
        self.sorted_conf_agent = []  # 最终结果：List：他车重要度排序
        self.dict_agent_lanelets = {}

    def extend2list(self, lanelet_network):
        '''为了适应接口。暂时修改
        '''
        conf_potential_lanelets = []
        conf_potential_points = []
        ids = self.dict_lanelet_conf_point.keys
        conf_points = self.dict_lanelet_conf_point.values

        for id, conf_point in zip(ids, conf_points):
            conf_lanlet = lanelet_network.find_lanelet_by_id(id)
            id_predecessors = conf_lanlet.predecessor
            # 排除没有父节点的情况
            if id_predecessors is not None:
                # 多个父节点
                for id_predecessor in id_predecessors:
                    conf_potential_lanelets.append(id_predecessor)
                    conf_potential_points.append(conf_point)
        return conf_potential_lanelets, conf_potential_points


def dense_cv(cv, interval=0.1):
    """
    将车道线的点加密为每隔 interval 米一个点。

    :param cv: 原始车道线点序列，格式为[[x1, y1], [x2, y2], ...]
    :param interval: 生成点的间隔，默认为0.1米
    :return: 加密后的点序列
    """
    new_cv = [cv[0]]  # 初始化加密后的点序列，包含起点
    for i in range(1, len(cv)):
        start_point = np.array(cv[i - 1])
        end_point = np.array(cv[i])
        segment_length = np.linalg.norm(end_point - start_point)

        # 在当前段上插入新的点
        num_points = int(segment_length // interval)

        if segment_length != 0:
            direction = (end_point - start_point) / segment_length
        else:
            direction = np.array([0.0, 0.0])  # 或者其他合理的默认值
            logging.warning("Segment length is zero. Setting direction to [0.0, 0.0].")
        for j in range(1, num_points + 1):
            new_point = start_point + direction * interval * j
            new_cv.append(new_point.tolist())

    new_cv.append(cv[-1])  # 最后一个点为终点
    return np.array(new_cv)


# 定义主函数
def level_k_planner(scenario, planning_problem_set, route, ego_vehicle = None):
    # 打印自车信息和障碍物信息
    print(f"Ego Vehicle ID: {ego_vehicle.obstacle_id}, Initial Position: {ego_vehicle.state_list[0].position}, ")

    # 使用 conf_lanelet_checker 函数查找冲突点
    lanelet_network = scenario.lanelet_network
    incoming_lanelet_id_sub = route.lanelet_ids[0]
    cl_info = conf_lanelet_checker(lanelet_network, incoming_lanelet_id_sub, 3, route)

    iinfo = IntersectionInfo(cl_info)
    iinfo.dict_parent_lanelet = potential_conf_lanelet_checkerv2(lanelet_network, cl_info)

    # ---------------- 运动规划 --------------
    # 计算车辆前进的参考轨迹。ref_cv： [n, 2]。参考轨迹坐标.
    ref_cv, ref_orientation, ref_s = get_route_frenet_line(route.lanelet_ids, lanelet_network)

    # 在[T, T+400]的时间进行规划
    s = distance_lanelet(ref_cv, ref_s, ref_cv[0, :], ego_vehicle.state_list[0].position)  # 计算自车的frenet纵向坐标
    s_list = [s]
    state_list = []
    state_list.append(ego_vehicle.state_list[0])

    dict_lanelet_agent = conf_agent_checker(lanelet_network, iinfo.dict_lanelet_conf_point, scenario)

    print('直接冲突车辆', dict_lanelet_agent)
    iinfo.dict_lanelet_agent = dict_lanelet_agent

    # 间接冲突车辆
    dict_lanelet_potential_agent = potential_conf_agent_checker(lanelet_network, iinfo.dict_lanelet_conf_point,
                                                                iinfo.dict_parent_lanelet, route, scenario)
    print('间接冲突车辆', dict_lanelet_potential_agent)
    iinfo.dict_lanelet_potential_agent = dict_lanelet_potential_agent

    # 如果没有冲突车辆，使用默认规划器
    if dict_lanelet_agent == {} and dict_lanelet_potential_agent == {}:
        return None, None, None, None, True

    # 冲突点排序
    iinfo.sorted_lanelet, iinfo.i_ego = sort_conf_point(ego_vehicle.state_at_time(0).position,
                                                        iinfo.dict_lanelet_conf_point,
                                                        ref_cv, ref_s)
    # 按照冲突点先后顺序进行决策。找车，给冲突车辆排序
    sorted_conf_agent = []
    dict_agent_lanelets = {}
    iinfo.dict_agent_goal_points = {}
    for i_lanelet in range(iinfo.i_ego, min(2, len(iinfo.sorted_lanelet))):
        lanelet_id = iinfo.sorted_lanelet[i_lanelet]
        # 直接冲突
        if lanelet_id in dict_lanelet_agent.keys():
            sorted_conf_agent.append(iinfo.dict_lanelet_agent[lanelet_id])
            dict_agent_lanelets[sorted_conf_agent[-1]] = [lanelet_id]
            goal_position = lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices[-1]  # 中心线的最后一个点作为目标点
            iinfo.dict_agent_goal_points[sorted_conf_agent[-1]] = goal_position
        else:
            # 查找父节点
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            for parent_lanelet_id in lanelet.predecessor:
                if parent_lanelet_id not in dict_lanelet_potential_agent.keys():
                    # 如果是None, 没有父节点，也会进入该循环
                    continue
                else:
                    sorted_conf_agent.append(iinfo.dict_lanelet_potential_agent[parent_lanelet_id])
                    if sorted_conf_agent[-1] not in dict_agent_lanelets.keys():
                        dict_agent_lanelets[sorted_conf_agent[-1]] = [parent_lanelet_id, lanelet_id]

                    # 获取冲突车辆对应的目标点，取父节点 lanelet 中心线的最后一个点
                    parent_lanelet = lanelet_network.find_lanelet_by_id(parent_lanelet_id)
                    goal_position = parent_lanelet.center_vertices[-1]  # 父节点中心线的最后一个点作为目标点
                    iinfo.dict_agent_goal_points[sorted_conf_agent[-1]] = goal_position
                    continue
    iinfo.sorted_conf_agent = sorted_conf_agent
    iinfo.dict_agent_lanelets = dict_agent_lanelets

    # 使用planing_problem的初始状态和目标作为障碍物车的初始状态和目标
    ego_vehicle_id = ego_vehicle.obstacle_id
    ego_lanelet = lanelet_network.find_lanelet_by_id(route.lanelet_ids[0])
    ego_lanelet_dict = {
        'left_bound': dense_cv(ego_lanelet.left_vertices),
        'right_bound': dense_cv(ego_lanelet.right_vertices),
        'center_line': dense_cv(ego_lanelet.center_vertices)
    }

    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    desired_velocity = (planning_problem.goal.state_list[0].velocity.start + planning_problem.goal.state_list[
        0].velocity.end) / 2

    ego_initial_state = State(
        x=ego_vehicle.prediction.trajectory.state_list[-1].position[0],
        y=ego_vehicle.prediction.trajectory.state_list[-1].position[1],
        yaw=ego_vehicle.prediction.trajectory.state_list[-1].yaw_rate,
        v=ego_vehicle.prediction.trajectory.state_list[-1].velocity
    )
    ego_goal_state = State(
        x=planning_problem.goal.state_list[0].position.center[0],
        y=planning_problem.goal.state_list[0].position.center[1],
        yaw=0,
        v=desired_velocity
    )
    ego_vehicle_base = VehicleBase(vehicle_id=ego_vehicle_id, initial_state=ego_initial_state,
                              goal_state=ego_goal_state, level=1, lane_lines=ego_lanelet_dict)

    # 选择两辆的车作为障碍物车
    obstacle_vehicles = iinfo.sorted_conf_agent[:2]
    agent_id_1 = obstacle_vehicles[0].obstacle_id
    agent_id_2 = obstacle_vehicles[1].obstacle_id

    other_vehicles = []
    other1_initial_state = State(
        x=obstacle_vehicles[0].prediction.trajectory.state_list[-1].position[0],
        y=obstacle_vehicles[0].prediction.trajectory.state_list[-1].position[1],
        yaw=obstacle_vehicles[0].prediction.trajectory.state_list[-1].yaw_rate,
        v=obstacle_vehicles[0].prediction.trajectory.state_list[-1].velocity
    )
    other1_goal_state = State(
        x=iinfo.dict_agent_goal_points[agent_id_1][0],
        y=iinfo.dict_agent_goal_points[agent_id_1][1],
        yaw=0,
        v=desired_velocity
    )
    o1_lanelet = lanelet_network.find_lanelet_by_id(iinfo.dict_agent_lanelets[agent_id_1][1])
    o1_lanelet_dict = {
        'left_bound': dense_cv(o1_lanelet.left_vertices),
        'right_bound': dense_cv(o1_lanelet.right_vertices),
        'center_line': dense_cv(o1_lanelet.center_vertices)
    }
    other1 = VehicleBase(vehicle_id=agent_id_1, initial_state=other1_initial_state, goal_state=other1_goal_state, level=0,
                            lane_lines=o1_lanelet_dict)
    other_vehicles.append(other1)

    other2_initial_state = State(
        x=obstacle_vehicles[1].prediction.trajectory.state_list[-1].position[0],
        y=obstacle_vehicles[1].prediction.trajectory.state_list[-1].position[1],
        yaw=obstacle_vehicles[1].prediction.trajectory.state_list[-1].yaw_rate,
        v=obstacle_vehicles[1].prediction.trajectory.state_list[-1].velocity
    )
    other2_goal_state = State(
        x=iinfo.dict_agent_goal_points[agent_id_2][0],
        y=iinfo.dict_agent_goal_points[agent_id_2][1],
        yaw=0,
        v=desired_velocity
    )
    o2_lanelet = lanelet_network.find_lanelet_by_id(iinfo.dict_agent_lanelets[agent_id_2][1])
    o2_lanelet_dict = {
        'left_bound': dense_cv(o2_lanelet.left_vertices),
        'right_bound': dense_cv(o2_lanelet.right_vertices),
        'center_line': dense_cv(o2_lanelet.center_vertices)
    }
    other2 = VehicleBase(vehicle_id=agent_id_2, initial_state=other2_initial_state, goal_state=other2_goal_state, level=0,
                            lane_lines=o2_lanelet_dict)
    other_vehicles.append(other2)


    # ego_initial_state = State(x=1.9432, y=-0.08365, yaw=0, v=12)
    # ego_goal_state = State(x=18, y=13, yaw=1.46, v=12)
    # ego_vehicle_base = VehicleBase(vehicle_id=ego_vehicle_id, initial_state=ego_initial_state,
    #                           goal_state=ego_goal_state, level=1, lane_lines=ego_lanelet_dict)
    #
    # # 他车
    # other_vehicles = []
    #
    # other1_initial_state = State(x=29.44335, y=0.6504, yaw=2.966356686481549, v=5)
    # other1_goal_state = State(x=1.342025, y=3.7669, yaw=3.14, v=5)
    # print("Available keys in dict_agent_lanelets:", iinfo.dict_agent_lanelets.keys())
    #
    # o1_lanelet = lanelet_network.find_lanelet_by_id(iinfo.dict_agent_lanelets[50234][0])
    # o1_lanelet_dict = {
    #     'left_bound': dense_cv(o1_lanelet.left_vertices),
    #     'right_bound': dense_cv(o1_lanelet.right_vertices),
    #     'center_line': dense_cv(o1_lanelet.center_vertices)
    # }
    # other1 = VehicleBase(vehicle_id=50234, initial_state=other1_initial_state, goal_state=other1_goal_state, level=0,
    #                      lane_lines=o1_lanelet_dict)
    # other_vehicles.append(other1)
    #
    # other2_initial_state = State(x=14.63675, y=11.5179, yaw=-1.8079873667339916, v=7)
    # other2_goal_state = State(x=28.47935, y=-3.23115, yaw=-0.25, v=7)
    # o2_lanelet = lanelet_network.find_lanelet_by_id(iinfo.dict_agent_lanelets[50236][0])
    # o2_lanelet_dict = {
    #     'left_bound': dense_cv(o2_lanelet.left_vertices),
    #     'right_bound': dense_cv(o2_lanelet.right_vertices),
    #     'center_line': dense_cv(o2_lanelet.center_vertices)
    # }
    # other2 = VehicleBase(vehicle_id=50236, initial_state=other2_initial_state, goal_state=other2_goal_state, level=0,
    #                      lane_lines=o2_lanelet_dict)
    # other_vehicles.append(other2)

    # 配置参数
    config = {
        'delta_t': 0.1,
        'max_step': 8,
        'max_simulation_time': 0.5,
        'computation_budget': 15000,
        'lamda': 0.92,
        'weight_avoid': 100,
        'weight_safe': 4,
        'weight_offroad': 100,  # 更新权重
        'weight_direction': 5,
        'weight_distance': 20,  # 更新权重
        'weight_velocity': 0.5,
        'weight_reverse': 5,  # 添加新的权重
        'weight_goal_distance': 0.08,
        'weight_time': 0.001
    }

    delta_t = config['delta_t']
    max_simulation_time = config['max_simulation_time']
    time = 0.0

    conf_dict = iinfo.dict_lanelet_conf_point
    conf_list = list(conf_dict.items())
    conf1_point = conf_list[0][-1]
    conf2_point = conf_list[1][-1]
    conf3_point = conf_list[2][-1]
    # print(conf1_point)
    # print(conf2_point)
    # print(conf3_point)

    planner = KLevelPlanner(config)

    while time < max_simulation_time:
        # 检查是否所有车辆都到达目标
        if ego_vehicle_base.is_get_target and all([v.is_get_target for v in other_vehicles]):
            print("所有车辆均到达目标点")
            break

        # 自车决策
        ego_action, _ = planner.planning(ego_vehicle_base, other_vehicles)
        # 自车状态更新
        ego_vehicle_base.update_state(ego_action[0], delta_t)

        # 打印自车动作
        print(f"时间 {time:.2f}s，自车动作：{ego_action[0]}")
        print(
            f"位置：({ego_vehicle_base.footprint.state_list[-1].x:.2f}, {ego_vehicle_base.footprint.state_list[-1].y:.2f})，速度：{ego_vehicle_base.footprint.state_list[-1].v:.2f}")

        # 他车决策和状态更新
        for other in other_vehicles:
            other_planner = KLevelPlanner(config)
            other_action, _ = other_planner.planning(other, [ego_vehicle_base] + [v for v in other_vehicles if v != other])
            other.update_state(other_action[0], delta_t)
            # other_vehicles[0].level = compute_vehicle_level(other_vehicles[0].state,ego_vehicle_base.state ,conf1_point)
            # other_vehicles[1].level = compute_vehicle_level(other_vehicles[1].state,ego_vehicle_base.state,conf3_point)
            other_vehicles[0].level = compute_vehicle_level(other_vehicles[0].state, other_vehicles[1].state,
                                                            ego_vehicle_base, conf1_point)
            other_vehicles[1].level = compute_vehicle_level(other_vehicles[1].state, other_vehicles[0].state,
                                                            ego_vehicle_base, conf3_point)

            # 打印他车动作
            print(
                f"时间 {time:.2f}s，车辆 {other.vehicle_id} 动作：{other_action[0]} level：{other.level} 速度: {other.state.v}")

        # 检查自车是否到达目标
        if ego_vehicle_base.is_get_target:
            print("自车到达目标点")
            break

        time += delta_t

    # 打印自车和他车的轨迹
    print("自车轨迹：")
    for state in ego_vehicle_base.footprint.state_list:
        print(f"位置：({state.x:.2f}, {state.y:.2f})，速度：{state.v:.2f}")

    for other in other_vehicles:
        print(f"车辆 {other.vehicle_id} 轨迹：")
        for state in other.footprint.state_list:
            print(f"位置：({state.x:.2f}, {state.y:.2f})，速度：{state.v:.2f}")

    # 提取自车轨迹的 x 和 y 坐标
    ego_x = [state.x for state in ego_vehicle_base.footprint.state_list]
    ego_y = [state.y for state in ego_vehicle_base.footprint.state_list]
    velocity = [state.v for state in ego_vehicle_base.footprint.state_list]
    yaw = [state.yaw for state in ego_vehicle_base.footprint.state_list]

    return ego_x, ego_y, velocity, yaw, False


if __name__ == '__main__':
    # read in scenario and planning problem set
    # path_scenario = "/home/yanliang/dataset/data/t_junction_test/ZAM_Tjunction-1_4_T-1.xml"
    path_scenario = "/home/yanliang/commonroad-geometric/projects/graph_rl_agents/predictive_traffic_rule_compliance/ZAM_Tjunction-1_75_T-1.xml"
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()

    level_k_planner(scenario, planning_problem_set, route)
