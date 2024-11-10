import itertools
from abc import abstractmethod, ABC
from typing import Union

import numpy as np

from crmonitor.common.helper import gather
from crmonitor.monitor.monitor_node import (
    MonitorNode,
    RuleMonitorNode,
    AllMonitorNode,
    ExistMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor, OutputType
from crmonitor.monitor.rule import (
    RuleNode,
    ExistNode,
    PredicateNode,
    AllNode,
    IOType,
)


class RuleTreeVisitor(ABC):
    @abstractmethod
    def visit_rule_node(self, rule_node: Union[RuleNode, RuleMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_all_node(self, all_node: Union[AllNode, AllMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_exist_node(self, exist_node: Union[ExistNode, ExistMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        pass


class MonitorCreationRuleTreeVisitor(RuleTreeVisitor):
    def __init__(self, dt, output_type=OutputType.STANDARD):
        self.dt = dt
        self.output_type = output_type

    def visit_rule_node(self, rule_node: RuleNode, *ctx):
        children = [c.visit(self, *ctx) for c in rule_node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(
            rule_node, self.dt, self.output_type
        )
        return RuleMonitorNode(rule_node.name, children, monitor)

    def visit_all_node(self, all_node: AllNode, *ctx):
        children = [c.visit(self, *ctx) for c in all_node.children]
        return AllMonitorNode(all_node.name, children)

    def visit_exist_node(self, exist_node: ExistNode, *ctx):
        children = [c.visit(self, *ctx) for c in exist_node.children]
        return ExistMonitorNode(exist_node.name, children)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return predicate_node


class EvaluationMonitorTreeVisitor(RuleTreeVisitor):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD):
        self.other_ids = tuple()
        self.use_boolean = use_boolean
        self.output_type = output_type

    def walk(self, node: MonitorNode, world, time_step, ego_vehicle, *ctx):
        self.other_ids = tuple()
        return node.visit(self, world, time_step, (ego_vehicle.id,), *ctx)

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        world = ctx[0]
        time_step = ctx[1]
        # Collect child_values
        assert (
            rule_node.monitor.dt == world.dt
        ), f"Monitor constructed with dt={rule_node.monitor.dt} but got world state with dt={world.dt}!"
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
        val = rule_node.update(time_step, list(child_values.items()))
        return val

    def _visit_quant_node(self, node, *ctx):
        world, time_step, other_ids = ctx[:3]
        all_ids = world.vehicle_ids_for_time_step(time_step)
        remaining_ids = tuple(set(all_ids).difference(other_ids))
        values = []
        selected_ids = []
        for i in remaining_ids:
            # Change
            try:
                ids = other_ids + (i,)
                val = node.monitors[i].visit(self, world, time_step, ids, *ctx[2:])
                values.append(val)
                selected_ids.append(ids)
            except:
                continue
        return values, selected_ids

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(all_node, *ctx)
        other_ids = ctx[2]
        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            all_node.last_selected = all_node.monitors[self.other_ids[-1]]
        else:
            val = 1.0
            self.other_ids = other_ids
            all_node.last_selected = None
        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(exist_node, *ctx)
        other_ids = ctx[2]
        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            exist_node.last_selected = exist_node.monitors[self.other_ids[-1]]
        else:
            val = -1.0
            self.other_ids = other_ids
            exist_node.last_selected = None
        return val

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        world, time_step, other_ids = ctx[:3]
        predicate_ids = gather(other_ids, predicate_node.agent_placeholders)
        if (
            self.use_boolean
            or predicate_node.io_type == IOType.INPUT
            and self.output_type == OutputType.OUTPUT_ROBUSTNESS
        ):
            value = predicate_node.evaluate_boolean(world, time_step, predicate_ids)
            value = 1.0 if value else -1.0
        else:
            value = predicate_node.evaluate_robustness(world, time_step, predicate_ids)
        return value


class PredicateCollectorMonitorTreeVisitor(RuleTreeVisitor):
    def _visit(self, node):
        r = []
        for c in node.children:
            r.extend(c.visit(self))
        return r

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        return self._visit(rule_node)

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        if all_node.last_selected is None:
            # Visit the prototype monitor
            val = all_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = all_node.last_selected.visit(self, *ctx)
        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        if exist_node.last_selected is None:
            # Visit the prototype monitor
            val = exist_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = exist_node.last_selected.visit(self, *ctx)
        return val

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.latest_value)]


class PredicateVisualizerMonitorTreeVisitor(RuleTreeVisitor):
    """
    Returns list of dictionaries, each dictionary mapping vehicle ids to a possibly nested dict of draw-parameters
    """

    def _split_context(self, ctx):
        idx = 6
        is_effective = ctx[idx] if len(ctx) > idx else True
        return ctx[:idx], is_effective

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        draw_functions_nested = [c.visit(self, *ctx) for c in rule_node.children]
        return list(itertools.chain(*draw_functions_nested))

    def _visit_quant_node(self, node, *ctx):
        ctx, is_effective_so_far = self._split_context(ctx)
        draw_functions_for_effective_node = []
        if node.last_selected is not None:
            draw_functions_for_effective_node = node.last_selected.visit(
                self, *ctx, True and is_effective_so_far
            )
        draw_functions_nested = [
            monitor.visit(self, *ctx, False)
            for i, monitor in node.monitors.items()
            if monitor != node.last_selected
        ]
        return (
            list(itertools.chain(*draw_functions_nested))
            + draw_functions_for_effective_node
        )

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        return self._visit_quant_node(all_node, *ctx)

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        return self._visit_quant_node(exist_node, *ctx)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        ctx, is_effective = self._split_context(ctx)

        (
            add_vehicle_draw_params,
            predicate_names2vehicle_ids2values,
            predicate_name2predicate_evaluator,
            world,
            time_step,
            visualization_config,
        ) = ctx

        pred_name = predicate_node.evaluator.predicate_name
        latest_vehicle_ids = predicate_node.latest_vehicle_ids

        config_entry_key = pred_name if pred_name in visualization_config else "default"
        config_obj = visualization_config.get(config_entry_key, {})
        show_non_effective_predicate_instances_for_vehicles = config_obj.get(
            "show_non_effective_predicate_instances_for_vehicles", []
        )

        if (
            not is_effective
            and latest_vehicle_ids
            not in show_non_effective_predicate_instances_for_vehicles
        ):
            return ()

        predicate_name2predicate_evaluator[pred_name] = predicate_node.evaluator

        return predicate_node.evaluator.visualize(
            latest_vehicle_ids,
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )


class ResetMonitorTreeVisitor(RuleTreeVisitor):
    def _visit(self, node, *ctx):
        for c in node.monitors.values():
            c.visit(self, *ctx)

    def visit_rule_node(self, rule_node: Union[RuleNode, RuleMonitorNode], *ctx):
        rule_node.reset()

    def visit_all_node(self, all_node: Union[AllNode, AllMonitorNode], *ctx):
        self._visit(all_node, *ctx)

    def visit_exist_node(self, exist_node: Union[ExistNode, ExistMonitorNode], *ctx):
        self._visit(exist_node, *ctx)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        pass
