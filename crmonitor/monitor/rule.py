import copy
import inspect
import re
import sys
from abc import ABCMeta, abstractmethod
from enum import Enum

# by setting __all__ in __init__.py, all relevant modules are imported
from crmonitor.predicates import *
from crmonitor.monitor.monitor_node import MonitorNode


def get_all_predicate_evaluators():
    modules = inspect.getmembers(sys.modules["crmonitor.predicates"], inspect.ismodule)
    classes = []
    for _, module in modules:
        classes += inspect.getmembers(module, inspect.isclass)
    classes = list(filter(lambda p: p[0][:4] == "Pred", classes))
    predicate_class_map = {cls.predicate_name: cls for name, cls in classes if len(name) > 4 and name[:4] == "Pred"}
    return predicate_class_map


class IOType(Enum):
    OUTPUT = "output"
    INPUT = "input"


def parse_rule(full_rule_str, config, name=None):
    full_predicate_pattern = re.compile(
        r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)?_(?P<agents>(_a(\d)+)+)))"
    )
    quantification_pattern = re.compile(
        r"^(?P<quant>[AE])\sa(?P<veh_id>\d+):\s\((?P<rule>.*)\)$"
    )
    subrule_pattern = re.compile(r"[AE]\sa\d+:\s\(.*\)")
    if name is None:
        name = full_rule_str
    m = quantification_pattern.match(full_rule_str)
    if m is not None:
        # Quantification on top level
        sub_rule_str = m["rule"]
        children = parse_rule(sub_rule_str, config, "g0")
        quantified_vehicle = int(m.group("veh_id"))
        if m.group("quant") == "E":
            node = ExistNode([children], quantified_vehicle, name)
        elif m.group("quant") == "A":
            node = AllNode([children], quantified_vehicle, name)
        else:
            raise ValueError()
    else:
        mod_rule_str = full_rule_str
        predicate_assignment = set()
        sub_rules = []
        m = subrule_pattern.search(mod_rule_str)
        while m is not None:
            mod_rule_str = (
                mod_rule_str[: m.start()]
                + f"g{len(sub_rules)}"
                + mod_rule_str[m.end() :]
            )
            sub_rule_str = m[0]
            sub_rules.append(parse_rule(sub_rule_str, config, f"g{len(sub_rules)}"))
            m = subrule_pattern.match(mod_rule_str)

        pred_matches = full_predicate_pattern.finditer(mod_rule_str)
        pred_evaluators = get_all_predicate_evaluators()
        for m in pred_matches:
            pred_basename = m.group("pred_name")
            agent_string = m.group("agents")
            agents = re.split(r"_a", agent_string)
            predicate_agent_placeholders = []
            for a in agents:
                if a != "":
                    predicate_agent_placeholders.append(int(a))
            evaluator = pred_evaluators[pred_basename]
            if m.group("io_type") is None:
                io_type = IOType.OUTPUT
                full_name = m.group("pred_name") + "_" + m.group("agents")
            else:
                io_type = IOType.INPUT
                full_name = m.group("pred_name") + "_" + m.group("agents") + "_i"
            assert evaluator is not None
            p = PredicateNode(
                full_name,
                predicate_agent_placeholders,
                evaluator(config["traffic_rules_param"]),
                io_type,
            )
            mod_rule_str = mod_rule_str.replace(m.group(0), p.name)
            predicate_assignment.add(p)
        node = RuleNode(sub_rules + list(predicate_assignment), mod_rule_str, name)
    return node


class VisitorNode(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def visit(self, visitor, **ctx):
        pass


class RuleNode(VisitorNode):
    def __init__(self, children, rule_str, name):
        self.children = children
        self.name = name
        self.rule_str = rule_str

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)


class AllNode(VisitorNode):
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle

    def visit(self, visitor, *ctx):
        return visitor.visit_all_node(self, *ctx)


class ExistNode(VisitorNode):
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle

    def visit(self, visitor, *ctx):
        return visitor.visit_exist_node(self, *ctx)


class PredicateNode(MonitorNode, VisitorNode):
    def __init__(self, full_name, agent_placeholders, evaluator, io_type=IOType.OUTPUT):
        assert len(agent_placeholders) == evaluator.arity, (
            f"The arity of the evaluator for {full_name} should be "
            f"{len(agent_placeholders)}, but is {evaluator.arity}!"
        )
        super().__init__(full_name)
        self.agent_placeholders = tuple(agent_placeholders)
        self.evaluator = evaluator
        self.io_type = io_type
        self.latest_value = None
        self.latest_vehicle_ids = None

    def evaluate_boolean(self, world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_boolean(world, time_step, vehicle_ids)
        self.latest_value = 1.0 if value else -1.0
        self.latest_vehicle_ids = tuple(vehicle_ids)
        return value

    def evaluate_robustness(self, world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )
        self.latest_value = value
        self.latest_vehicle_ids = tuple(vehicle_ids)
        return value

    def visit(self, visitor, *ctx):
        return visitor.visit_predicate_node(self, *ctx)

    @property
    def base_name(self):
        return self.evaluator.predicate_name

    @property
    def num_dependencies(self):
        return len(self.agent_placeholders)

    def __eq__(self, o) -> bool:
        return self.name == o.name and self.agent_placeholders == o.agent_placeholders

    def __hash__(self) -> int:
        return hash((self.name, self.agent_placeholders))

    def copy(self):
        return copy.copy(self)

    def reset(self):
        self.latest_value = None
        self.latest_vehicle_ids = None
