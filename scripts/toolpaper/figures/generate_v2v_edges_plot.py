import sys, os; sys.path.insert(0, os.getcwd())

import shutil
from pathlib import Path

from commonroad_geometric.common.logging import stdout
from commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import *
from commonroad_geometric.dataset.extraction.traffic.temporal_traffic_extractor import (TemporalTrafficExtractor,
                                                                                        TemporalTrafficExtractorOptions)
from commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractor, TrafficExtractorOptions
from commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import VehicleFilterPreprocessor
from commonroad_geometric.rendering.color.color import Color
from commonroad_geometric.rendering.plugins.cameras.follow_vehicle_camera import FollowVehicleCamera
from commonroad_geometric.rendering.plugins.cameras.global_map_camera import GlobalMapCamera
from commonroad_geometric.rendering.plugins.implementations import RenderLaneletNetworkPlugin, RenderTrafficGraphPlugin
from commonroad_geometric.rendering.plugins.obstacles.render_obstacle_plugin import RenderObstaclePlugin
from commonroad_geometric.rendering.traffic_scene_renderer import TrafficSceneRenderer, TrafficSceneRendererOptions
from commonroad_geometric.rendering.types import RenderParams
from commonroad_geometric.rendering.viewer.pyglet.gl_viewer_2d import GLViewerOptions
from commonroad_geometric.simulation.interfaces.static.scenario_simulation import (ScenarioSimulation,
                                                                                   ScenarioSimulationOptions)

INPUT_SCENARIO = Path('data/highway_test')
EXPORT_DIR = Path('outputs/toolpaper/v2v')
EDGE_DRAWERS = [VoronoiEdgeDrawer, FullyConnectedEdgeDrawer, KNearestEdgeDrawer]
EDGE_DISTANCE_THRESHOLD = 25.0
EXTRACT_TEMPORAL = False
RENDERER_SIZE = (400, 400)
SCREENSHOT_RATE = 5
HD_RESOLUTION_MULTIPLIER = 5.0
CAMERA_FOLLOW = False
VIEW_RANGE = 50.0


def generate_v2v_edges_plot(
    enable_hd: bool,
    enable_screenshots: bool = True,
    scenario_dir: Path = INPUT_SCENARIO
) -> None:
    if enable_screenshots:
        shutil.rmtree(EXPORT_DIR, ignore_errors=True)

    RENDERERS: dict[str, TrafficSceneRenderer] = {}
    for edge_drawer_cls in EDGE_DRAWERS:
        renderer = TrafficSceneRenderer(
            options=TrafficSceneRendererOptions(
                viewer_options=GLViewerOptions(
                    window_width=RENDERER_SIZE[0],
                    window_height=RENDERER_SIZE[1],
                    window_scaling_factor=HD_RESOLUTION_MULTIPLIER if enable_hd else 1.0,
                    transparent_screenshots=True,
                ),
                camera=FollowVehicleCamera(view_range=VIEW_RANGE) if CAMERA_FOLLOW else GlobalMapCamera(),
                plugins=[
                    RenderLaneletNetworkPlugin(),
                    RenderObstaclePlugin(),
                    RenderTrafficGraphPlugin(
                        node_radius=0.3,
                        node_linewidth=0.01,
                        node_fillcolor=Color((0.0, 0.9, 0.0, 1.0))
                    ),
                ],
                fps=1
            )
        )
        RENDERERS[edge_drawer_cls.__name__] = renderer

    preprocessor = VehicleFilterPreprocessor()
    # preprocessor >>= LaneletNetworkSubsetPreprocessor(radius=500.0)
    # preprocessor >>= DepopulateScenarioPreprocessor(5)
    # preprocessor >>= ValidTrajectoriesFilterer()
    scenario_iterator = ScenarioIterator(
        directory=scenario_dir,
        preprocessor=preprocessor,
    )

    print(f"Enjoying {len(scenario_iterator.scenario_paths)} scenarios from {scenario_dir=}")
    print(f"Preprocessing strategy: {preprocessor}")

    frame_count: int = 0
    for scenario_bundle in scenario_iterator:
        simulation = ScenarioSimulation(
            initial_scenario=scenario_bundle.preprocessed_scenario,
            options=ScenarioSimulationOptions(
                backup_current_scenario=False,
                backup_initial_scenario=False
            )
        )

        extractors = {}

        for edge_drawer_cls in EDGE_DRAWERS:
            if EXTRACT_TEMPORAL:
                sub_extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=edge_drawer_cls(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
                extractor = TemporalTrafficExtractor(
                    traffic_extractor=sub_extractor,
                    options=TemporalTrafficExtractorOptions(
                        collect_num_time_steps=15,
                        collect_skip_time_steps=0,
                        return_incomplete_temporal_graph=True
                    )
                )
            else:
                extractor = TrafficExtractor(
                    simulation=simulation,
                    options=TrafficExtractorOptions(
                        edge_drawer=edge_drawer_cls(dist_threshold=EDGE_DISTANCE_THRESHOLD),
                        ignore_unassigned_vehicles=False
                    )
                )
            extractors[edge_drawer_cls.__name__] = extractor

        simulation.start()

        for time_step, scenario in simulation:
            for name, extractor in extractors.items():
                data = extractor.extract(time_step=time_step)
                renderer = RENDERERS[name]

                capture_screenshot = enable_screenshots and frame_count % SCREENSHOT_RATE == 0
                if capture_screenshot:
                    renderer.screenshot(
                        output_file=f"{EXPORT_DIR}/v2v_edges_plot_{frame_count}_{name}.png",
                        queued=True
                    )

                simulation.render(
                    renderers=[renderer],
                    return_frames=False,
                    render_params=RenderParams(
                        time_step=time_step,
                        scenario=scenario,
                        data=data
                    )
                )

            stdout(f"Enjoying {scenario.scenario_id} (timestep {time_step}/{simulation.final_time_step})")

            frame_count += 1
