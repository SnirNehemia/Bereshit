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

    # Name: 2025-07-23_T_21-06-03
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: no cannibalism allowed (father/child cannot eat other)
    # Config changes: None
    # Result: carnivores goes up a little and then down with herbivores. all died at step 9092

    # Name: 2025-07-23_T_23-04-59
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: None
    # Config changes: CHANCE_TO_HERBIVORE = 0.5 (was 0.3)
    # Result: carnivores goes up a little and then down with herbivores. all died at step 12116

    # Name: 2025-07-24_T_16-56-23
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: None
    # Config changes: CHANCE_TO_HERBIVORE = 0.8 (was 0.5)
    # Result: carnivore goes up and then down when herbivores starts to die rapidly

    # Name: 2025-07-24_T_21-05-17
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: added death_causes_dict to understand why everyone dies around step ~4000
    # Config changes: None
    # Result: carnivore goes up and then down when herbivores starts to die rapidly.
    #         I understood that most die from age since dt=2.5 and 8000<=max_age<=10000 so it ok.

    # Name: 2025-07-25_T_00-53-07
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: fixed negative mutated traits.
    #               now there shouldn't be set_trace() exception that makes some creatures not move.
    # Config changes: MAX_AGE = 40000 (was 10000)
    # Result: need more children in general, once 1st generation died, all gradually dies

    # Name: 2025-07-25_T_11-33-22
    # Goal: carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: None
    # Config changes: REPRODUCTION_ENERGY = 3000 (was 5000)
    # Result: Amazing! we see positive slope!

    # Name: 2025-07-25_T_16-27-00
    # Goal: carnivores survival
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Code changes: red edge color for carnivores, green for herbivores
    # Result: Amazing! we see carnivores stable until herbivores almost gone.
    #         Then all carnivores died and herbivores flourished.

    # Name: 2025-07-26_T_00-42-51
    # Goal: carnivores survival, we need more herbivores first according to last run to get balance.
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Config change: CHANCE_TO_HERBIVORE = 0.9 (was 0.8)
    # Result: Stopped in the middle (50% took 9 hours)

    # Name: 2025-07-26_T_14-22-47
    # Goal: carnivores survival, make them stable at around ~200 creatures
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Config changes: REPRODUCTION_ENERGY = 5000 (was 3000)
    # Result: Stopped in the middle, but we have a video (46% took 5 and a half hours)
    #         At the end we had carnivores and herbivores.

    # Name: 2025-07-26_T_19-59-00
    # Goal: carnivores survival, make them stable at around ~200 creatures
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Config changes: NUM_CREATURES = 200 (was 500), MAX_NUM_CREATURES = 500 (was 1500)
    # Result: First good run. Completed after 2 hours. only herbivores left

    # Name: 2025-07-26_T_23-43-08
    # Goal: carnivores survival, make them stable at around ~200 creatures
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Config changes: CHANCE_TO_HERBIVORE = 0.8 (was 0.9)
    # Result: only herbivores survived.

    # Name: 2025-07-27_T_01-11-42
    # Goal: carnivores survival, make them stable at around ~200 creatures
    # Num iterations: 305 frames, 100000 steps
    # Code changes: None
    # Config changes: CHANCE_TO_HERBIVORE = 0.85 (was 0.8)
    # Result: Stopped in the middle. Only herbivores survived.

    # ---------------------------------------------------------------------------------------
    # Reconstruct statistics logs
    outputs_folder_relative_path = Path(r"outputs")
    timestamp = "2025-07-26_T_23-43-08"
    statistics_logs = load_statistics_logs_obj(timestamp=timestamp,
                                               outputs_folder_relative_path=outputs_folder_relative_path)

    # Plot things
    # statistics_logs.plot_creatures_lifespan_vs_step_matrix()
    # statistics_logs.plot_creatures_nutrition_vs_step_matrix()
    # statistics_logs.plot_creatures_lifespan_graphs()
    # statistics_logs.plot_creatures_nutrition_graphs()
    statistics_logs.plot_num_herbivores_vs_num_carnivores_per_step(to_show=True)

    print([f"{key}: {len(value)}" for key, value in statistics_logs.death_causes_dict.items()])

