import importlib.resources as pkg_resources
import logging
import shelve
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, lru_cache
from pathlib import Path
from typing import Optional, Set, Union

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle

import crmonitor
from crmonitor.common.helper import (create_other_vehicles_param, load_yaml, )
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import (Vehicle, DynamicObstacleVehicle, CurvilinearStateManager, PredicateCache,
                                      ControlledVehicle, )


@lru_cache(maxsize=None)
def get_world_config():
    with pkg_resources.path(crmonitor, "config.yaml") as config_path:
        config = load_yaml(config_path)
    return config


@dataclass
class World:
    vehicles: Set[Vehicle]
    road_network: RoadNetwork
    scenario: Optional[Scenario] = None
    cache: Union[None, shelve.Shelf, dict] = None

    def _warn_persistent_cache(self):
        if len(self.controlled_vehicle_ids) > 0 and isinstance(self.cache, shelve.Shelf):
            warnings.warn(
                "Using controlled vehicles with persistent caching may result in inconsistent caches and is therfore discouraged!")

    def __post_init__(self):
        self._warn_persistent_cache()

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles.add(vehicle)
        self._warn_persistent_cache()

    def add_controlled_vehicle(self, scenario: Scenario, obstacle: DynamicObstacle):
        config = get_world_config()
        params = config.get("road_network_param")
        road_network = RoadNetwork(scenario.lanelet_network, params)
        others_params = create_other_vehicles_param(config.get("other_vehicles_param"))
        self.augment_state_acceleration_jerk(scenario.dt, obstacle)
        self.vehicles.add(
            ControlledVehicle(
                obstacle.obstacle_id,
                others_params,
                obstacle.obstacle_shape,
                road_network,
                obstacle.initial_state,
                obstacle.obstacle_type
            ))

    def remove_vehicle_states(self, time_step):
        for vehicle in self.vehicles:
            if isinstance(vehicle, ControlledVehicle):
                vehicle.remove_state(time_step)

    @classmethod
    def create_from_scenario(
        cls, scenario: Scenario, config=None, road_network=None, cache_dir=None,  # is_ego: bool = False
    ):
        if config is None:
            config = get_world_config()
        if road_network is None:
            params = config.get("road_network_param")
            road_network = RoadNetwork(scenario.lanelet_network, params)
        else:
            road_network = road_network
        others_params = create_other_vehicles_param(config.get("other_vehicles_param"))
        vehicles = set()
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"{scenario.scenario_id}"
            cache = shelve.open(str(cache_file), writeback=True)
        else:
            cache = {}
        for obs in scenario.dynamic_obstacles:
            # Skip obstacles that go out of the road
            # if any(map(lambda a: len(a) == 0, obs.prediction.shape_lanelet_assignment.values())):
            #     continue
            cls.augment_state_acceleration_jerk(scenario.dt, obs)
            vehicles.add(
                ControlledVehicle(
                    obs.obstacle_id,
                    others_params,
                    obs.obstacle_shape,
                    road_network,
                    obs.initial_state,
                    obs.obstacle_type
                )
            )
        # print(f'Vehicle ids: \n {[vehicle.id for vehicle in vehicles]}')
        # if is_ego:
        #     vehicles.add(
        #         ControlledVehicle(
        #             obs.obstacle_id,
        #             others_params,
        #             obs.obstacle_shape,
        #             road_network,
        #             obs.initial_state,
        #             obs.obstacle_type
        #         )
        #     )
        # else:
        #     curvi_cache, predicate_dict = cache.setdefault(
        #         str(obs.obstacle_id), (dict(), defaultdict(partial(defaultdict, dict)))
        #     )
        #     vehicles.add(
        #         DynamicObstacleVehicle(
        #             obs,
        #             CurvilinearStateManager(road_network, curvi_cache),
        #             others_params,
        #             PredicateCache(predicate_dict),
        #         ),
        #     )
        return cls(vehicles, road_network, scenario, cache)

    @property
    def controlled_vehicle_ids(self) -> Set[int]:
        return {vehicle.id for vehicle in self.vehicles if isinstance(vehicle, ControlledVehicle)}

    def vehicle_ids_for_time_step(self, time_step: int):
        return [v.id for v in self.vehicles if v.is_valid(time_step)]

    @staticmethod
    def augment_state_acceleration_jerk(dt, obs):
        accelerations = (
            np.diff(
                [
                    s.velocity
                    for s in [obs.initial_state] + obs.prediction.trajectory.state_list
                ]
            )
            / dt
        ).tolist()
        obs.initial_state.acceleration = accelerations[0]
        if len(accelerations) >= 2:
            jerk = (np.diff(accelerations) / dt).tolist()
            accelerations += accelerations[-1:]
            jerk += [jerk[-1], 0]
            obs.initial_state.jerk = jerk[0]
        else:
            jerk = [None] * 2
        for a, j, state in zip(
            accelerations[1:], jerk[1:], obs.prediction.trajectory.state_list
        ):
            state.acceleration = a
            if j is not None:
                state.jerk = j

    def vehicle_by_id(self, id) -> Optional[Vehicle]:
        for veh in self.vehicles:
            if veh.id == id:
                break
        else:
            logging.warning(f"Vehicle with ID {id} not found!")
            veh = None
        return veh

    @property
    def dt(self):
        if self.scenario is not None:
            return self.scenario.dt
        else:
            return 0.1

    def __del__(self):
        if isinstance(self.cache, shelve.Shelf):
            logging.info("Cache close!")
            self.cache.close()
