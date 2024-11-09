import numpy as np
from commonroad.scenario.state import State, CustomState
from typing import List

def predict_constant_velocity(initial_state: State, dt: float, num_steps: int = 20) -> List[State]:
    """
    使用恒速运动学模型预测未来状态

    Args:
        initial_state: 初始状态，包含position, orientation, velocity, acceleration
        dt: 时间步长（秒）
        num_steps: 预测步数

    Returns:
        预测状态列表
    """
    # 获取初始状态值
    position = initial_state.position  # [x, y]
    velocity = initial_state.velocity  # 标量速度
    orientation = initial_state.orientation  # 航向角
    acceleration = initial_state.acceleration
    time_step = initial_state.time_step

    # 存储预测结果
    predicted_states = []

    # 预测未来状态
    for i in range(num_steps):
        # 计算新的时间步
        new_time_step = time_step + i + 1

        # 使用恒速模型计算新位置
        # dx = v * cos(θ) * dt
        # dy = v * sin(θ) * dt
        dx = velocity * np.cos(orientation) * dt
        dy = velocity * np.sin(orientation) * dt
        new_position = np.array([
            position[0] + dx,
            position[1] + dy
        ])

        # 创建新的状态
        new_state = CustomState(
            time_step=new_time_step,
            position=new_position,
            orientation=orientation,  # 保持恒定方向
            velocity=velocity,  # 保持恒定速度
            acceleration=acceleration  # 保持恒定加速度
        )

        # 更新当前位置用于下一步预测
        position = new_position

        # 添加到预测列表
        predicted_states.append(new_state)

    return predicted_states