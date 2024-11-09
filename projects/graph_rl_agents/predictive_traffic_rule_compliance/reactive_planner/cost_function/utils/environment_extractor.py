import torch
from typing import List, Dict, Any, Tuple
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.observer.implementations.flattened_graph_observer import FlattenedGraphObserver
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions, TrafficExtractionParams

def extract_environment_info(
    ego_vehicle_simulation: EgoVehicleSimulation,
    observer: FlattenedGraphObserver,
    device: torch.device,
    evaluation_time_steps: List[int]
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """
    提取指定评估时间步的环境数据。

    参数：
        ego_vehicle_simulation: EgoVehicleSimulation 实例。
        observer: 用于提取观测的观察者。
        device: 设备（CPU 或 GPU）。
        evaluation_time_steps: 将要提取观测的时间步列表。

    返回：
        environment_data_dict: 一个字典，映射时间步到 CommonRoadData。
        environment_info: 包含静态环境信息的字典。
    """

    # 提取静态环境信息
    environment_info = {
        'ego_vehicle_params_b': ego_vehicle_simulation.ego_vehicle.parameters.b,
        'ego_vehicle_id': ego_vehicle_simulation.ego_vehicle.obstacle_id,
        'ego_vehicle_simulation': ego_vehicle_simulation,  # 需要在后续提取数据时使用
    }

    # 为需要的时间步提取环境数据
    environment_data_dict = {}

    # 获取模拟器和 traffic_extractor
    simulation = ego_vehicle_simulation.simulation

    # 初始化 traffic_extractor
    traffic_extractor_options = TrafficExtractorOptions(
        edge_drawer=observer.edge_drawer,  # 使用 observer 中的 edge_drawer
        feature_computers=observer.feature_computers,
        # 根据需要设置其他选项
    )
    traffic_extractor = TrafficExtractor(
        simulation=simulation,
        options=traffic_extractor_options,
    )

    # 提取每个评估时间步的 CommonRoadData
    for time_step in evaluation_time_steps:
        # 创建提取参数
        extraction_params = TrafficExtractionParams(
            ego_vehicle=ego_vehicle_simulation.ego_vehicle,
            device=device,
        )

        # 提取数据
        commonroad_data = traffic_extractor.extract(
            time_step=time_step,
            params=extraction_params
        )

        # 存储数据
        environment_data_dict[time_step] = commonroad_data

    return environment_data_dict, environment_info
