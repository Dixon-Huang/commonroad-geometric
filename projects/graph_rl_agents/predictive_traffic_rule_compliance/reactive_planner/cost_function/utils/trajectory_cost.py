# trajectory_cost.py

import numpy as np
import torch
import copy
import math
from commonroad.scenario.state import State
from stable_baselines3.common.utils import obs_as_tensor
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

def compute_trajectory_cost(
    trajectory,
    environment_data,
    environment_info,
    value_function,
    observer,
    device,
    desired_speed=None,
    desired_d=0.0,
    desired_s=None,
):
    try:
        costs = 0.0
        # 初始化状态列表，用于模拟ego车辆沿轨迹行驶
        state_list = []

        # 模拟ego车辆沿轨迹行驶
        num_points = len(trajectory.cartesian.x)
        indices = _get_evaluation_indices(num_points)
        indices_set = set(indices)  # 为了快速查找

        value_list = []
        for i in range(len(trajectory.cartesian.x)):
            # 转换坐标系
            position_rear = np.array([trajectory.cartesian.x[i], trajectory.cartesian.y[i]])
            orientation = trajectory.cartesian.theta[i]
            position_center = position_rear + environment_info['ego_vehicle'].parameters.b * np.array(
                [math.cos(orientation), math.sin(orientation)]
            )

            state = State(
                position=position_center,
                velocity=trajectory.cartesian.v[i],
                acceleration=trajectory.cartesian.a[i],
                orientation=orientation,
                time_step=environment_info['current_time_step'] + i + 1,  # 根据需要调整时间步
                yaw_rate=0.0,
                slip_angle=0.0
            )

            state_list.append(state)

            if i in indices_set:
                # 在指定的时间步提取obs并计算value
                obs_tensor = _extract_obs_at_state(
                    state,
                    environment_data,
                    environment_info,
                    observer,
                    device
                )
                with torch.no_grad():
                    value = value_function.predict_values(obs_tensor)
                value_list.append(value)

        # 计算成本
        value_array = np.vstack([v.detach().cpu().numpy() for v in value_list])
        costs -= 1e4 * (np.sum(5 * value_array) +
                        (50 * value_array[-1]) +
                        (100 * value_array[int(len(value_array) / 2) - 1]))

        # 加入其他成本计算
        # 加速度成本
        costs += np.sum((5 * trajectory.cartesian.a) ** 2)
        # 速度成本
        if desired_speed is not None:
            costs += np.sum((5 * (trajectory.cartesian.v - desired_speed)) ** 2) + \
                     (50 * (trajectory.cartesian.v[-1] - desired_speed) ** 2) + \
                     (100 * (trajectory.cartesian.v[int(len(trajectory.cartesian.v) / 2)] - desired_speed) ** 2)
        if desired_s is not None:
            costs += np.sum((0.25 * (desired_s - trajectory.curvilinear.s)) ** 2) + \
                     (20 * (desired_s - trajectory.curvilinear.s[-1])) ** 2
        # 距离成本
        costs += np.sum((0.25 * (desired_d - trajectory.curvilinear.d)) ** 2) + \
                 (20 * (desired_d - trajectory.curvilinear.d[-1])) ** 2
        # 方向成本
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + \
                 (5 * np.abs(trajectory.curvilinear.theta[-1])) ** 2

    except Exception as e:
        logger.error(f"Error in cost function calculation: {e}")

    return float(costs)

def _extract_obs_at_state(
    ego_state,
    environment_data,
    environment_info,
    observer,
    device
):
    # 创建一个新的数据对象，包含当前ego车辆的状态
    data = copy.deepcopy(environment_data)
    data.ego_vehicle_state = ego_state

    # 使用 observer 提取 obs
    obs = observer.observe(
        data=data,
        ego_vehicle_simulation=None  # 不需要模拟器
    )
    with torch.no_grad():
        obs_tensor = obs_as_tensor(obs, device)
    return obs_tensor

def _get_evaluation_indices(num_points):
    idx_mid = int(round(0.5 * (num_points - 1)))
    idx_end = num_points - 1

    indices = [idx_mid, idx_end]
    # 去除重复索引并排序
    indices = sorted(set(indices))
    return indices

@staticmethod
def compute_costs_parallel(
    trajectories: List[commonroad_rp.trajectories.TrajectorySample],
    cost_function: 'RobustnessCostFunctionSimplified',
    max_workers: Optional[int] = None
) -> List[float]:
    """
    Compute costs for a list of trajectories in parallel.

    Args:
        trajectories: List of trajectory samples.
        cost_function: An instance of RobustnessCostFunctionSimplified.
        max_workers: Maximum number of worker processes.

    Returns:
        List of costs corresponding to the trajectories.
    """
    costs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare data that can be serialized
        serialized_environment_data_dict = cost_function.environment_data_dict
        serialized_environment_info = cost_function.environment_info
        value_function_state = cost_function.value_function.state_dict()
        device = cost_function.device
        observer_state = cost_function.observer.__dict__

        futures = [
            executor.submit(
                _compute_trajectory_cost_parallel,
                traj,
                serialized_environment_data_dict,
                serialized_environment_info,
                value_function_state,
                observer_state,
                device,
                cost_function.desired_speed,
                cost_function.desired_d,
                cost_function.desired_s
            )
            for traj in trajectories
        ]
        for future in concurrent.futures.as_completed(futures):
            costs.append(future.result())
    return costs

