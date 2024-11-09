import os
import re
from typing import List, Iterator, Tuple
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader


def is_scenario_id(scenario_id: str) -> bool:
    pattern = r"^[a-zA-Z]{3}_[a-zA-Z0-9]+-\d+_\d+(_[TS]-\d)?$"
    return re.match(pattern, scenario_id)


def is_coop_scenario_id(scenario_id: str) -> bool:
    pattern = r"^C-[a-zA-Z]{3}_[a-zA-Z0-9]+-\d+_\d+_[TS]-\d+$"
    pattern2 = r"^C-[a-zA-Z]{3}_[a-zA-Z0-9]+-\d+_\d+$"
    first_match = re.match(pattern, scenario_id)
    second_match = re.match(pattern2, scenario_id)
    return first_match or second_match


def is_interactive_scenario_id(scenario_id: str) -> bool:
    pattern = r"^[a-zA-Z]{3}_[a-zA-Z0-9]+-\d+_\d+_I-[\d\_\-]+$"
    return re.match(pattern, scenario_id)


def is_scenario_file(path: str) -> bool:
    normpath = os.path.normpath(path)
    is_file = os.path.isfile(normpath)

    filename = os.path.basename(normpath)
    is_xml = filename.endswith(".xml")

    scenario_id = filename.replace(".xml", "")
    is_normal_id = is_scenario_id(scenario_id)
    is_coop_id = is_coop_scenario_id(scenario_id)
    return is_file and is_xml and (is_normal_id or is_coop_id)


def is_interactive_scenario_dir(path: str) -> bool:
    normpath = os.path.normpath(path)
    dirname = os.path.basename(normpath)
    int_id = is_interactive_scenario_id(dirname)
    cr_filepath = os.path.join(path, f"{dirname}.cr.xml")
    contains_cr_file = os.path.isfile(cr_filepath)
    return int_id and contains_cr_file


def _search_scenarios(path: str) -> List[str]:
    normpath = os.path.normpath(path)

    if is_scenario_file(normpath):
        return [normpath]

    scenario_paths = []
    for pth, subdirs, files in os.walk(normpath):
        for file in files:
            fullpath = os.path.join(pth, file)
            if is_scenario_file(fullpath):
                scenario_paths.append(fullpath)

    return scenario_paths


def _search_interactive_scenarios(path: str) -> List[str]:
    normpath = os.path.normpath(path)

    if is_interactive_scenario_dir(normpath):
        return [path]

    scenario_paths = []
    for pth, dirs, files in os.walk(normpath):
        if not dirs:
            pass

        for dirname in dirs:
            fullpath = os.path.join(pth, dirname)
            if is_interactive_scenario_dir(fullpath):
                scenario_paths.append(fullpath)

    return scenario_paths


def scenario_iterator_non_interactive(path: str) -> Iterator[Tuple[Scenario, PlanningProblemSet]]:
    """
    Glob the given path for scenarios and return a generator
    object for loading scenarios and planning problem sets.
    """
    scenario_paths = _search_scenarios(path)

    # Create and return a generator for loading scenarios
    return (
        CommonRoadFileReader(scenario_path).open()
        for scenario_path in scenario_paths
    )


def scenario_iterator_interactive(path: str) -> Iterator[str]:
    """
    Glob the given path for sceanarios and return a generator
    object for the paths to interactive scenario folders.
    """
    return iter(_search_interactive_scenarios(path))
