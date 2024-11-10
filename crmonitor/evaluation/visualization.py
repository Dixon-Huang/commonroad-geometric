from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd

from commonroad.common.util import Interval
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.scenario import Scenario
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from crmonitor.common.helper import merge_dicts_recursively
from crmonitor.predicates.base import BasePredicateEvaluator


EGO_VEHICLE_DRAW_PARAMS = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {"shape": {"rectangle": {"facecolor": "yellow"}}}
        }
    }
}


def plot_rule_robustness_course(
        rule_robustness_course: List[Tuple[int, float]],
        ax,
        plot_limits: Tuple[float, float],
        rules: List[str]
):
    np_rule_robustness_course = np.array(rule_robustness_course)
    rob_values = np_rule_robustness_course[:, 1]
    ax.plot(rob_values, "b-")
    ax.plot(np.where(rob_values < 0.0, rob_values, np.nan), "rx")
    ax.plot(np.where(rob_values >= 0.0, rob_values, np.nan), "g.")
    ax.set_xlim([np_rule_robustness_course[0, 0], np_rule_robustness_course[-1, 0]])
    ax.set_ylim(plot_limits)
    ax.grid(True)
    ax.set_ylabel(f"robustness of rule {','.join(rules)}")
    ax.set_xlabel("time step")


def plot_predicate_bar_chart(
    predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ax,
    bar_chart_plot_limits: Tuple[float, float],
):
    df = pd.DataFrame.from_dict(
        {
            predicate_name: {
                str(vehicle_ids): values
                for vehicle_ids, values in vehicle_ids2values.items()
            }
            for predicate_name, vehicle_ids2values in predicate_names2vehicle_ids2values.items()
        }
    )

    # we use a different color map, as default one produces non-distinguishable colors for different bars
    cmap = plt.get_cmap("turbo")
    numbers_for_bars = np.linspace(0, 1, num=len(df.columns), endpoint=False)
    ax = df.plot.barh(
        rot=0,
        ax=ax,
        width=1.0,
        edgecolor="black",
        linewidth=0.5,
        color=cmap(numbers_for_bars),
        xlim=bar_chart_plot_limits,
    )
    ax.set_ylabel("vehicle ids")
    ax.set_xlabel("predicate robustness")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))  # place legend to the right


def _create_axes(
    scenario_fig_size: Tuple[float, float],
    nr_rules: int,
    flag_plot_predicate_bar_chart: bool,
    flag_plot_rule_robustness_course: bool,
    flag_rule_conjunction: bool
):
    bar_plots = []
    rob_plots = []

    if flag_plot_predicate_bar_chart or flag_plot_rule_robustness_course:
        fig = plt.figure(constrained_layout=True,
                         figsize=(scenario_fig_size[0], scenario_fig_size[1]*(1+nr_rules)))
        if flag_rule_conjunction:
            n_rows = 2
        else:
            n_rows = nr_rules + 1
        n_cols = sum([flag_plot_predicate_bar_chart, flag_plot_rule_robustness_course])
        gs = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)
        scenario_ax = fig.add_subplot(gs[0, :])
        rob_plot_index = 0
        for r in range(n_rows - 1):
            if flag_plot_predicate_bar_chart:
                bar_plots.append(fig.add_subplot(gs[r + 1, 0]))
                rob_plot_index = 1
            if flag_plot_rule_robustness_course:
                rob_plots.append(fig.add_subplot(gs[r + 1, rob_plot_index]))
    else:
        plt.figure(figsize=scenario_fig_size)
        scenario_ax = plt.gca()

    return scenario_ax, bar_plots, rob_plots


def _plot_scenario_legend(
    predicate_name2predicate_evaluator: Dict[str, BasePredicateEvaluator],
    scenario_fig_size: Tuple[float, float],
):
    width, _ = scenario_fig_size
    figsize = (width, width / 4)
    num_predicates = len(predicate_name2predicate_evaluator)
    fig, (axes_row_1, axes_row_2) = plt.subplots(
        figsize=figsize, nrows=2, ncols=num_predicates
    )
    fig.suptitle("Legend: predicate visualization in scenario", fontsize=14)
    for ax1, ax2, (pred_name, pred_evaluator) in zip(
        axes_row_1, axes_row_2, predicate_name2predicate_evaluator.items()
    ):
        ax1.text(0.1, 0.5, pred_name, fontsize=12)
        ax1.axis("off")
        pred_evaluator.plot_predicate_visualization_legend(ax2)


def plot_rule_visualization(scenario: Scenario,
                            ego_vehicle_id: int,
                            time_step: int,
                            rule_evaluator_list,
                            visualization_config: Dict[str, any],
                            scenario_fig_size: Tuple[float, float] = (10., 2.),
                            bar_chart_plot_limits: Tuple[float, float]=(-1.0, 1.0),
                            rule_robustness_course_plot_limits: Tuple[float, float]=(-1.0, 1.0),
                            flag_plot_predicate_bar_chart: bool = True,
                            flat_plot_rule_robustness_course: bool = True,
                            scenario_plot_limits: Union[List[Union[int, float]], None] = None,
                            flag_rule_conjunction: bool = False,
                            plot_scenario_legend: Optional[bool]=None):
    """
    Plotting the rule evaluation result
    :param scenario: the CommonRoad scenario to be visualized
    :param ego_vehicle_id: id of ego vehicle (the vehicle to be controlled)
    :param time_step: the time step of the current scenario
    :param rule_evaluator_list: precreated list of rule evaluators
    :param visualization_config: user-defined configuration of visualization
    :param scenario_fig_size: size of scenario plot
    :param bar_chart_plot_limits: the plot limits of x-axis
    :param rule_robustness_course_plot_limits: the plot limits of x-axis
    :param flag_plot_predicate_bar_chart: flag of whether the bar chart needs to be plotted
    :param flat_plot_rule_robustness_course: flag of whether the robustness curve needs to be plotted
    :param scenario_plot_limits: the plot limits of scenario,
    :param flag_rule_conjunction: whether consider the conjunction of rules or separately calculate them
    :plot_scenario_legend: whether the legend for the scenario visualization should be plotted. If None, it is
    plotted for the first time-step only
    """

    nr_rules = len(rule_evaluator_list)

    general_draw_params = {
        "time_begin": time_step,
        "dynamic_obstacle": {
            "show_label": True,
            "vehicle_shape": {
                "occupancy": {"shape": {"rectangle": {"facecolor": "#90ee90"}}}
            },
        },
    }

    vehicle2draw_params = {}
    pred_result_dict = {}
    rule_result_dict = {}
    rule_name_list = []
    all_predicate_name2predicate_evaluator = {}
    # Hint: plotting further stuff on the scenario only works after renderer.render() was called; therefore, the
    #   predicates need to return functions instead of directly plotting
    all_draw_functions = []
    for i in range(nr_rules):
        rule_evaluator_list[i].update()
        predicate_name2predicate_evaluator, pred_result, rule_result, draw_functions = rule_evaluator_list[i].visualize_predicates(vehicle2draw_params,
                                                                               visualization_config)
        all_predicate_name2predicate_evaluator.update(predicate_name2predicate_evaluator)
        all_draw_functions += draw_functions
        pred_result_dict[rule_evaluator_list[i]._rule.name] = pred_result  # merge the dict
        rule_result_dict[rule_evaluator_list[i]._rule.name] = rule_result
        rule_name_list.append(rule_evaluator_list[i]._rule.name)

    if plot_scenario_legend or plot_scenario_legend is None and time_step == 0:
        _plot_scenario_legend(all_predicate_name2predicate_evaluator, scenario_fig_size)

    scenario_ax, bar_chart_axs, robustness_course_axs = _create_axes(scenario_fig_size, nr_rules,
                                                                     flag_plot_predicate_bar_chart,
                                                                     flat_plot_rule_robustness_course,
                                                                     flag_rule_conjunction)

    renderer = MPRenderer(ax=scenario_ax, plot_limits=scenario_plot_limits)

    if flag_rule_conjunction:
        if flag_plot_predicate_bar_chart:
            pred_conjunct_dict = defaultdict(dict)
            for _, pred_result_sep in pred_result_dict.items():
                for veh_ids, rob_pairs in pred_result_sep.items():
                    pred_conjunct_dict[veh_ids].update(rob_pairs)
            plot_predicate_bar_chart(pred_conjunct_dict, bar_chart_axs[0], bar_chart_plot_limits)

        if flat_plot_rule_robustness_course:
            # conjunction of all rules, i.e., the min of the robustness is calculated
            rule_rob_list = [r for _, rule_rob in rule_result_dict.items() for r in rule_rob]
            rule_conjunct_list = [min(time_rob[1]) for time_rob in groupby(rule_rob_list,
                                                                           lambda rule_rob_list: rule_rob_list[0])]
            plot_rule_robustness_course(rule_conjunct_list, robustness_course_axs[0],
                                        rule_robustness_course_plot_limits, rule_name_list)

    else:
        i = 0
        for rule in rule_name_list:
            if flag_plot_predicate_bar_chart:
                plot_predicate_bar_chart(pred_result_dict[rule], bar_chart_axs[i], bar_chart_plot_limits)

            if flat_plot_rule_robustness_course:
                plot_rule_robustness_course(rule_result_dict[rule], robustness_course_axs[i],
                                            rule_robustness_course_plot_limits, [rule])
            i += 1
    # after vehicle2draw_params is determined, draw the scenarios
    scenario.lanelet_network.draw(renderer, draw_params=general_draw_params)

    # plotting scenario and obstacles
    if scenario_plot_limits:
        plot_veh_ids = [obs.obstacle_id for obs in scenario.obstacles_by_position_intervals(
                [Interval(scenario_plot_limits[0], scenario_plot_limits[1]),
                 Interval(scenario_plot_limits[2], scenario_plot_limits[3])], time_step=time_step)]
    else:
        plot_veh_ids = [obs.obstacle_id for obs in scenario.obstacles]
    for i in plot_veh_ids:
        if i != ego_vehicle_id:
            draw_params = vehicle2draw_params.get(i, {})
            scenario.obstacle_by_id(i).draw(
                renderer,
                draw_params=merge_dicts_recursively(general_draw_params, draw_params),
            )

    scenario.obstacle_by_id(ego_vehicle_id).draw(renderer,
                                                 draw_params=merge_dicts_recursively(general_draw_params,
                                                                                     EGO_VEHICLE_DRAW_PARAMS), )
    renderer.render()

    for f in all_draw_functions:
        f(renderer)
