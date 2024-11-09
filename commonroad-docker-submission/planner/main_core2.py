import os.path
import hydra
from commonroad.common.solution import Solution, CommonRoadSolutionWriter

# Load planner module
from iterator import scenario_iterator_interactive, scenario_iterator_non_interactive, _search_interactive_scenarios
from projects.graph_rl_agents.predictive_traffic_rule_compliance.main_interactive import motion_planner_interactive


def save_solution(solution: Solution, path: str) -> None:
    """
    Save the given solution to the given path.
    """
    return CommonRoadSolutionWriter(solution).write_to_file(
        output_path=path,
        overwrite=True,
        pretty=True
    )


# Run Main Process
if __name__ == "__main__":
    scenario_dir = "/commonroad/scenarios"
    solution_dir = "/commonroad/solutions"

    config_path = "./"
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name="config")

    # solve the second half of interactive scenarios
    interactive_paths = _search_interactive_scenarios(scenario_dir)
    n_interactive_scenarios = len(interactive_paths)
    last_half_scenario_path = interactive_paths[int(n_interactive_scenarios/2):]

    for scenario_path in last_half_scenario_path:
        print(f"Processing scenario {os.path.basename(scenario_path)} ...")
        try:
            solution = motion_planner_interactive(scenario_path, cfg)
            save_solution(solution, solution_dir)
            print('-' * 20, scenario_path, 'solved', '-' * 20)
        except Exception as e:
            print('-' * 20, f'Cannot solve scenario {scenario_path}: {str(e)}', '-' * 20)
            continue
