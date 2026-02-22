import matplotlib.pyplot as plt

from input.codes import repos_utils, sim_config
from statistics_logs import StatisticsLogs

from matplotlib import use

use('TkAgg')


def get_run_folder_path(timestamp: str, outputs_folder_name: str):
    project_folder = repos_utils.fetch_directory()
    datestamp = timestamp.split('_T')[0]
    run_folder_path = project_folder.joinpath(outputs_folder_name).\
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


if __name__ == '__main__':
    # Reconstruct statistics logs
    outputs_folder_name = "outputs"
    timestamp = "2025-07-27_T_01-11-42"
    # timestamp = "2026-02-22_T_02-30-07"

    statistics_logs = load_statistics_logs_obj(timestamp=timestamp,
                                               outputs_folder_name=outputs_folder_name)

    # Plot things
    # statistics_logs.plot_creatures_statistics(timestamp=timestamp, stat_trait='energy')
    # statistics_logs.plot_num_food_sources(timestamp=timestamp)
    # statistics_logs.plot_creatures_lifespan_vs_step_matrix(timestamp=timestamp)
    # statistics_logs.plot_creatures_nutrition_vs_step_matrix(timestamp=timestamp) # plot on top of previous graph now.
    # statistics_logs.plot_creatures_lifespan_graphs(timestamp=timestamp)
    # statistics_logs.plot_creatures_nutrition_graphs(timestamp=timestamp)
    statistics_logs.plot_num_herbivores_vs_num_carnivores_per_step(timestamp=timestamp)
    # statistics_logs.plot_death_causes_statistics(timestamp=timestamp)
    # statistics_logs.plot_nutrition_rate_graph(timestamp=timestamp)
    plt.show()
