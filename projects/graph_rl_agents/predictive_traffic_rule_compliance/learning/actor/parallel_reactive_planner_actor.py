import torch.multiprocessing as mp
from typing import List, Optional
import numpy as np
import torch
from projects.graph_rl_agents.predictive_traffic_rule_compliance.learning.actor.reactive_planner_actor import \
    ReactivePlannerActor, ReactivePlannerActorOptions
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


def worker(index, actor, input_queue, output_queue):
    while True:
        task = input_queue.get()
        if task is None:
            break
        ego_vehicle_simulation, value_function = task
        result = actor.step([ego_vehicle_simulation], value_function)
        output_queue.put((index, result))


class ParallelReactivePlannerActor:
    def __init__(self, options: ReactivePlannerActorOptions, num_envs: int):
        self.num_envs = num_envs
        self.options = options
        self.actors = [ReactivePlannerActor(options) for _ in range(num_envs)]

        mp.set_start_method('spawn', force=True)  # 使用 'spawn' 方法来启动进程
        self.input_queues = [mp.Queue() for _ in range(num_envs)]
        self.output_queue = mp.Queue()

        self.processes = []
        for i in range(num_envs):
            p = mp.Process(target=worker, args=(i, self.actors[i], self.input_queues[i], self.output_queue))
            p.start()
            self.processes.append(p)

    def step(self, ego_vehicle_simulations: List[EgoVehicleSimulation], value_function) -> Optional[List[np.ndarray]]:
        for i, ego_sim in enumerate(ego_vehicle_simulations):
            self.input_queues[i].put((ego_sim, value_function))

        results = [None] * self.num_envs
        for _ in range(self.num_envs):
            idx, result = self.output_queue.get()
            results[idx] = result

        if any(result is None for result in results):
            return None  # 如果任何环境需要重置，返回 None

        return [result[0] for result in results]  # 解包单元素数组

    def __del__(self):
        for queue in self.input_queues:
            queue.put(None)  # 发送终止信号
        for p in self.processes:
            p.join()