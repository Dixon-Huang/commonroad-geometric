import copy
from enum import Enum
from typing import List, Tuple

import rtamt
from rtamt import Language
from rtamt.evaluator.stl.online_evaluator import STLOnlineEvaluator

from crmonitor.monitor.rule import IOType, RuleNode


class OutputType(Enum):
    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


class RtamtStlMonitor:
    specs = {}
    """
    Represents single formalized STL rule
    """
    @staticmethod
    def _reconstruct_logic_formula(logic_formula, predicates):
        replacements = {"~": "not"}
        for el in replacements.keys():
            logic_formula = logic_formula.replace(el, replacements[el])
        # Workaround for rtamt when working with output-robustness and input vacuity
        mod_formula = logic_formula
        for pred in predicates:
            mod_formula = mod_formula.replace(pred[0].name, f"({pred[0].name} >= 0)")
        return mod_formula

    @staticmethod
    def construct_monitor(formula, output_type: OutputType, predicates, dt) -> rtamt.STLSpecification:
        logic_formula = RtamtStlMonitor._reconstruct_logic_formula(formula, predicates)
        monitor = rtamt.STLDiscreteTimeSpecification(
            semantics=output_type, language=Language.PYTHON
        )
        for var, io_type in predicates:
            monitor.declare_var(var.name, "float")
            if io_type == IOType.INPUT:
                monitor.set_var_io_type(var.name, "input")
            else:
                monitor.set_var_io_type(var.name, "output")
        monitor.declare_var("out", "float")

        monitor.iosem = output_type
        monitor.unit = "ms"
        monitor.spec = f"out = {logic_formula}"
        monitor.set_sampling_period(dt * 1000.0, 'ms')
        monitor.parse()
        monitor.pastify()

        return monitor

    @classmethod
    def create_from_rule_node(cls, rule_node: RuleNode, dt: float, output_type=OutputType.STANDARD):
        predicates = [(c, c.io_type if hasattr(c, "io_type") else IOType.OUTPUT) for c in rule_node.children]
        return cls(rule_node.rule_str, predicates, dt, output_type)

    def __init__(self, rule_str, predicates, dt, output_type=OutputType.STANDARD):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self.dt = dt
        spec = self.specs.get((rule_str, output_type, dt))
        if spec is None:
            spec = self.specs.setdefault((rule_str, output_type, dt), self.construct_monitor(rule_str, output_type, predicates, dt))
            self.specs[(rule_str, output_type, dt)] = spec
        # Flat copy spec and only recreate the online evaluator to avoid parsing the rule.
        self._monitor = copy.copy(spec)
        self._monitor.online_evaluator = STLOnlineEvaluator(self._monitor)
        self._monitor.top.accept(self._monitor.online_evaluator)
        self._monitor.reseter.node_monitor_dict = self._monitor.online_evaluator.node_monitor_dict
        self._monitor.reset()

    def reset_monitor(self):
        self._monitor.reset()

    def evaluate_monitor_online(self, time_step: int, predicates: List[Tuple[str, float]]):
        time = time_step * self.dt * 1000.0
        rob = self._monitor.update(time, predicates)
        return rob

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._monitor.reset()