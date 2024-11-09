import threading
from queue import Queue
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
        input_queue.task_done()

class MultithreadingReactivePlannerActor:
    def __init__(self, options: ReactivePlannerActorOptions, num_envs: int):
        self.num_envs = num_envs
        self.options = options
        self.actors = [ReactivePlannerActor(options) for _ in range(num_envs)]

        self.input_queues = [Queue() for _ in range(num_envs)]
        self.output_queue = Queue()

        self.threads = []
        for i in range(num_envs):
            t = threading.Thread(target=worker, args=(i, self.actors[i], self.input_queues[i], self.output_queue))
            t.daemon = True  # 设置为守护线程，这样主程序退出时它们会自动结束
            t.start()
            self.threads.append(t)

    def step(self, ego_vehicle_simulations: List[EgoVehicleSimulation], value_function) -> Optional[List[np.ndarray]]:
        for i, ego_sim in enumerate(ego_vehicle_simulations):
            self.input_queues[i].put((ego_sim, value_function))

        results = [None] * self.num_envs
        for _ in range(self.num_envs):
            idx, result = self.output_queue.get()
            results[idx] = result
            self.output_queue.task_done()

        if any(result is None for result in results):
            return None  # 如果任何环境需要重置，返回 None

        return [result[0] for result in results]  # 解包单元素数组

    def __del__(self):
        for queue in self.input_queues:
            queue.put(None)  # 发送终止信号
        for queue in self.input_queues:
            queue.join()  # 等待所有任务完成
        self.output_queue.join()
        for t in self.threads:
            t.join()  # 等待所有线程结束