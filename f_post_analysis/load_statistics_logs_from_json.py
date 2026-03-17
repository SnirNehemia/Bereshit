import matplotlib.pyplot as plt

from matplotlib import use

from a_utils import repos_utils
from b_basic.sim_config import sim_config
from e_logs.statistics_logs import StatisticsLogs

use('TkAgg')


def get_run_folder_path(timestamp: str, outputs_folder_name: str):
    project_folder = repos_utils.fetch_directory()
    datestamp = timestamp.split('_T')[0]
    run_folder_path = project_folder.joinpath(outputs_folder_name). \
        joinpath(datestamp).joinpath(timestamp)
    return run_folder_path


def load_statistics_logs_obj(timestamp: str, outputs_folder_name: str):
    # Get run folder
    run_folder_path = get_run_folder_path(
        timestamp=timestamp, outputs_folder_name=outputs_folder_name)

    # Load config
    config_path = run_folder_path.joinpath(timestamp + '_config.yaml')
    sim_config.load_config(config_name=config_path,
                           folder_full_path=run_folder_path)

    # Load statistics_logs
    statistics_logs_path = run_folder_path.joinpath(timestamp + '_statistics_logs.json')
    statistics_logs = StatisticsLogs.from_json(filepath=statistics_logs_path)

    return statistics_logs


def seconds_to_hhmmss(seconds):
    # Ensure the input is an integer for correct operations
    seconds = int(seconds)

    # Calculate hours, minutes, and remaining seconds
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    # Use an f-string for formatting with leading zeros
    return f"{hours:02}:{minutes:02}:{seconds:02}"


if __name__ == '__main__':
    # Reconstruct statistics logs
    # herbivore chance = 0.85
    # outputs_folder_name = r"outputs\steps_40k"
    # timestamps = ["2026-03-06_T_12-14-07", "2026-03-06_T_12-56-57",
    #               "2026-03-06_T_13-20-56", "2026-03-06_T_13-49-55",
    #               "2026-03-06_T_14-18-20"]

    # herbivore chance = 0.85
    # outputs_folder_name = r"outputs\steps_100k"
    # timestamps = ["2026-03-06_T_16-48-27", "2026-03-06_T_17-44-36"]

    # herbivore chance = 0.75
    outputs_folder_name = r"outputs\steps_155k"
    timestamps = ["2026-03-07_T_23-37-24",
                  "2026-03-08_T_06-41-04", "2026-03-08_T_11-05-23",
                  "2026-03-08_T_18-43-45", "2026-03-08_T_20-19-23"]

    # herbivore chance = 0.65
    # outputs_folder_name = r"outputs\steps_155k"
    # timestamps = []

    for timestamp in timestamps:
        statistics_logs = load_statistics_logs_obj(timestamp=timestamp,
                                                   outputs_folder_name=outputs_folder_name)

        # print things
        print(f'{timestamp}: run time {seconds_to_hhmmss(statistics_logs.total_time)}')
        print(f"T: {statistics_logs.num_creatures_per_step[0]}, "
              f"H: {statistics_logs.num_herbivores_per_step[0]}, "
              f"C: {statistics_logs.num_carnivores_per_step[0]}, "
              f"T/H: {round(statistics_logs.num_herbivores_per_step[0] / statistics_logs.num_creatures_per_step[0], 2)}")
        print('-----')

        # Plot things
        # statistics_logs.plot_creatures_statistics(timestamp=timestamp, stat_trait='energy')
        # statistics_logs.plot_num_food_sources(timestamp=timestamp)
        # statistics_logs.plot_creatures_lifespan_vs_step_matrix(timestamp=timestamp)
        # statistics_logs.plot_creatures_nutrition_vs_step_matrix(timestamp=timestamp) # plot on top of previous graph now.
        # statistics_logs.plot_creatures_lifespan_graphs(timestamp=timestamp)
        # statistics_logs.plot_creatures_nutrition_graphs(timestamp=timestamp)
        # statistics_logs.plot_num_herbivores_vs_num_carnivores_per_step(timestamp=timestamp)
        statistics_logs.plot_death_causes_statistics(timestamp=timestamp)
        # statistics_logs.plot_nutrition_rate_graph(timestamp=timestamp)
        plt.show()
