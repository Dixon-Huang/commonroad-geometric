import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer


def visualize_scenario(scenario, planning_problem_set, start_time_step=0, end_time_step=40, time_step_interval=1, fig_size=(25, 10)):
    """
    Visualizes the scenario and planning problem set over a range of time steps.

    :param scenario: The scenario object to visualize.
    :param planning_problem_set: The planning problem set to visualize.
    :param start_time_step: The starting time step for visualization (default is 0).
    :param end_time_step: The ending time step for visualization (default is 40).
    :param time_step_interval: The interval of time steps to visualize (default is 1).
    :param fig_size: The size of the figure (default is (25, 10)).
    """
    plt.rcParams["figure.max_open_warning"] = 50
    fig, ax = plt.subplots(figsize=fig_size)

    for i in range(start_time_step, end_time_step, time_step_interval):
        ax.clear()  # clear the current contents of the plot
        rnd = MPRenderer(ax=ax)
        # set time step in draw_params
        rnd.draw_params.time_begin = i
        # plot the scenario at different time step
        scenario.draw(rnd)
        # plot the planning problem set
        planning_problem_set.draw(rnd)
        rnd.render()
        plt.pause(0.1)  # pause to update the plot

    plt.show()


if __name__ == "__main__":
    # Example usage:
    file_path = "example_scenarios/ZAM_Over-1_1.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    visualize_scenario(scenario, planning_problem_set)
