from pathlib import Path

from input.codes import repos_utils
from input.codes.config import load_config
from input.codes.physical_model import load_physical_model


def get_run_folder_relative_path(timestamp: str, outputs_folder_relative_path: Path):
    datestamp = timestamp.split('_T')[0]
    run_folder_relative_path = outputs_folder_relative_path.joinpath(datestamp).joinpath(timestamp)
    return run_folder_relative_path


def load_config_and_physical_model(timestamp: str, outputs_folder_relative_path: Path):
    run_folder_relative_path = get_run_folder_relative_path(
        timestamp=timestamp, outputs_folder_relative_path=outputs_folder_relative_path)
    config_path = run_folder_relative_path.joinpath(timestamp + '_config.yaml')
    physical_model_path = run_folder_relative_path.joinpath(timestamp + '_physical_model.yaml')

    _ = load_config(yaml_relative_path=config_path)
    _ = load_physical_model(yaml_relative_path=physical_model_path)


def load_statistics_logs_obj(timestamp: str, outputs_folder_relative_path: Path):
    # Load config and physical model
    load_config_and_physical_model(
        timestamp=timestamp, outputs_folder_relative_path=outputs_folder_relative_path)

    # Load and reconstruct statistics_logs
    project_folder = repos_utils.fetch_directory()
    run_folder_relative_path = get_run_folder_relative_path(
        timestamp=timestamp, outputs_folder_relative_path=outputs_folder_relative_path)
    statistics_logs_path = project_folder.joinpath(run_folder_relative_path). \
        joinpath(timestamp + '_statistics_logs.json')

    # Load and reconstruct statistics logs
    from statistics_logs import StatisticsLogs

    statistics_logs = StatisticsLogs.from_json(filepath=statistics_logs_path)

    return statistics_logs


if __name__ == '__main__':
    # ----------------------- Eating creatures didn't work -------------------------------
    # Name: 2025-06-21_T_18-55-17
    # Num iterations: longer run (200 frames, 20500 steps)
    # Code changes:
    # added num_herbivores_per_step and carnivores to statistics logs
    # (notice num_herbivores_per_step and num_carnivores_per_step are swapped)
    # Config changes:
    # REPRODUCTION_ENERGY = 4000 (was 5000)
    # GRASS_GROWTH_CHANCE = 0.1 (was 0.5)
    # INIT_MAX_AGE = 10000 (was 60000)

    # ----------------------- Eating creatures work -------------------------------
    # Name: 2025-06-22_T_00-15-50
    # Num iterations: same as above (200 frames, 20500 steps)
    # Code changes: fixed problem that carnivores didn't eat at all
    # Result: All died pretty fast (after 7000 steps)
    # Found bug: all creatures could "eat" grass & creature (without gaining energy)
    #                because didn't check digest_dict[food_type] == 0 in eat_food function before eat function

    # Name: 2025-06-22_T_00-57-58
    # Num iterations: same as above (200 frames, 20500 steps)
    # Code changes: fixed the bug found above
    # Config changes: REPRODUCTION_ENERGY = 3000 (was 4000)

    # Name: 2025-06-22_T_01-33-52
    # Num iterations: much longer run (200 frames, 102500 steps)
    # Result: all died around step 25000

    # Name: 2025-06-22_T_12-00-30
    # Num iterations: shorter run to check for carnivores reproduction (15 frames, 10000 steps)
    # Code changes: carnivore that eat creature now get its energy also
    # Result: There are carnivore children! but they didn't survive until the end

    # Name: 2025-06-22_T_12-35-39
    # Num iterations: longer run to check for carnivores children stability (200 frames, 102500 steps)
    # Config changes: back to REPRODUCTION ENERGY = 5000 (was 3000)
    # Result: all died around step 12000

    # Name: 2025-06-22_T_14-22-27
    # Goal: more carnivores at the beginning to achieve carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: added num frames and num steps to statistics_logs
    # Config changes: CHANCE_TO_HERBIVORE = 0.3 (was 0.5), less iterations (not needed right now)
    # Result: all died around step 20000

    # Reconstruct statistics logs
    outputs_folder_relative_path = Path(r"saved_outputs")
    timestamp = "2025-06-22_T_14-22-27"
    statistics_logs = load_statistics_logs_obj(timestamp=timestamp,
                                               outputs_folder_relative_path=outputs_folder_relative_path)

    # Plot things
    statistics_logs.plot_creatures_lifespan_vs_step_matrix()
    statistics_logs.plot_creatures_nutrition_vs_step_matrix()
    statistics_logs.plot_creatures_lifespan_graphs()
    statistics_logs.plot_creatures_nutrition_graphs(to_show=True)
    statistics_logs.plot_num_herbivores_vs_num_carnivores_per_step(to_show=True)
