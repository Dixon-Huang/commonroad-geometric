from abc import ABC, abstractmethod
import copy
import logging
from typing import Optional
from commonroad_rp.cost_function import CostFunction
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
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
                 ego_vehicle_simulation: EgoVehicleSimulation = None,
                 robustness_types: Optional[list] = None,
                 ) -> None:
        super(BaselineCostFunction, self).__init__()
        # Target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s
        self.ego_vehicle_simulation = ego_vehicle_simulation
        # Weights
        self.w_a = 5  # Acceleration weight
        self.w_r = 5  # Robustness weight
        # List of robustness types (rules)
        if robustness_types is None:
            # Default to all the rules you provided
            # self.robustness_types = ['R_G1', 'R_G2', 'R_G3', 'R_G4', 'R_I1', 'R_I2', 'R_I3', 'R_I4', 'R_I5']
            self.robustness_types = ['R_G1']
        else:
            self.robustness_types = robustness_types

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        try:
            costs = 0.0
            saved_state = self.ego_vehicle_simulation.save_state()
            ego_vehicle_simulation = self.ego_vehicle_simulation

            # Compute rule robustness at specific points along the trajectory
            robustness_list, indices = self._calculate_robustness(trajectory, ego_vehicle_simulation)
            robustness_array = np.array(robustness_list)

            # Since we are minimizing cost, and higher robustness is better,
            # we subtract the robustness from the cost (or equivalently, add negative robustness)
            # Multiply by weight and scaling factor
            costs -= 1e4 * (np.sum(5 * robustness_array) + \
                            (50 * robustness_array[-1]) + \
                            (100 * robustness_array[int(len(robustness_array) / 2)]))

            # Other cost calculations
            # Acceleration costs
            costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)

            # Velocity costs
            if self.desired_speed is not None:
                costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                         (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                         (100 * (trajectory.cartesian.v[
                                     int(len(trajectory.cartesian.v) / 2)] - self.desired_speed) ** 2)
            if self.desired_s is not None:
                costs += np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2) + \
                         (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2
            # Distance costs
            costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                     (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
            # Orientation costs
            costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                    5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        except Exception as e:
            logger.error(f"Error in cost function calculation: {e}")

        finally:
            # Restore the original state
            self.ego_vehicle_simulation.restore_state(saved_state, indices)

        return float(costs)

    def _calculate_robustness(self, trajectory: commonroad_rp.trajectories.TrajectorySample,
                              ego_vehicle_simulation: EgoVehicleSimulation):
        """
        Compute the sum of rule robustness at specific points along the trajectory.
        """
        try:
            ego_vehicle = ego_vehicle_simulation.ego_vehicle
            robustness_list = []
            # Initialize RuleEvaluator with all the rules
            ego_vehicle_world_state = ego_vehicle_simulation.simulation.world_state.vehicle_by_id(-1)
            rule_evaluators = []
            for rule in self.robustness_types:
                rule_evaluator = RuleEvaluator.create_from_config(
                    ego_vehicle_simulation.simulation.world_state,
                    ego_vehicle_world_state,
                    rule
                )
                rule_evaluators.append(rule_evaluator)

            num_points = len(trajectory.cartesian.x)
            indices = self._get_evaluation_indices(num_points)
            indices_set = set(indices)  # For faster lookup
            initial_time_step = copy.deepcopy(ego_vehicle_simulation.current_time_step)

            for i in range(len(trajectory.cartesian.x) - 1):
                if i + 1 in indices_set:
                    # Transform coordinates
                    position_rear = np.array([trajectory.cartesian.x[i + 1], trajectory.cartesian.y[i + 1]])
                    orientation = trajectory.cartesian.theta[i + 1]
                    position_center = position_rear + ego_vehicle.parameters.b * np.array(
                        [math.cos(orientation), math.sin(orientation)])
                    time_step = initial_time_step + 1 + i

                    state = InitialState(
                        position=position_center,
                        velocity=trajectory.cartesian.v[i + 1],
                        acceleration=trajectory.cartesian.a[i + 1],
                        orientation=orientation,
                        time_step=time_step,
                        yaw_rate=0.0,
                        slip_angle=0.0
                    )

                    ego_vehicle.set_state(state)
                    ego_vehicle_simulation.simulation.jump_to_time_step(time_step)

                    total_robustness = 0.0
                    for evaluator in rule_evaluators:
                        robustness_series = evaluator.evaluate()
                        robustness = robustness_series[-1]
                        # 如果robustness为inf，说明规则不适用，不计入总robustness
                        if robustness == float('inf'):
                            continue
                        total_robustness += robustness
                    robustness_list.append(total_robustness)
            return robustness_list, indices

        except Exception as e:
            logger.error(f"Error in robustness calculation: {e}")
            return []

    def _get_evaluation_indices(self, num_points):
        """
        Get the indices corresponding to the specific points where robustness is evaluated.
        """
        idx_q1 = int(round(0.25 * (num_points - 1)))
        idx_mid = int(round(0.5 * (num_points - 1)))
        idx_q3 = int(round(0.75 * (num_points - 1)))
        idx_end = num_points - 1

        indices = [
            # idx_q1,
            idx_mid,
            # idx_q3,
            idx_end
                   ]
        # Remove duplicates and sort
        indices = sorted(set(indices))
        return indices
