import abc
import logging
import math
from typing import List, Tuple, Dict, Callable
import numpy as np
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World

from commonroad.visualization.renderer import IRenderer

logger = logging.getLogger(__name__)

MAX_LONG_DIST = 200.0
MAX_LAT_DIST = 20.0


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap):
        self.config = config
        self.scale = config.setdefault("scale_rob", True)
        self.eps = 1e-5

    # todo: decouple the scaler
    def _scale(self, x, max_value):
        return np.clip(x / max_value, -1.0, 1.0) if self.scale else x

    def _scale_speed(self, x):
        return self._scale(x, 250.0 / 3.6)

    def _scale_acc(self, x):
        return self._scale(x, 10.5)

    def _scale_lon_dist(self, x):
        return self._scale(x, MAX_LONG_DIST)

    def _scale_lat_dist(self, x):
        return self._scale(x, MAX_LAT_DIST)

    def _scale_angle(self, x):
        return self._scale(x, math.pi)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        pass

    def evaluate_robustness_with_cache(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        )
        if value is None:
            logger.debug(
                "Evaluating predicate %s , t=%d, ids=%s",
                self.predicate_name,
                time_step,
                vehicle_ids_tuple,
            )
            value = self.evaluate_robustness(world, time_step, vehicle_ids)
            vehicle.predicate_cache.set_robustness(
                time_step, self.predicate_name, vehicle_ids_tuple[1:], value
            )
        return value

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ) -> Tuple[Callable[[IRenderer], None], ...]:
        """
        Overwrite this function for visualizing a predicate in a certain way within the scenario plot.
        """
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        return ()

    def _gather_predicate_values_to_plot(
        self,
        vehicle_ids: List[int],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        predicate_names2vehicle_ids2values[self.predicate_name][
            tuple(vehicle_ids)
        ] = self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)
