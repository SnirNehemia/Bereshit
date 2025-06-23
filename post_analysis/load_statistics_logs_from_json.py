import numpy as np

from input.codes.config import load_config
from input.codes.physical_model import load_physical_model
import matplotlib.pyplot as plt
from matplotlib import use

use('TkAgg')
# deleted from top of StaticTraits and Creature so no need for config here:
# brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
# Brain = getattr(brain_module, 'Brain')

# Load config
config_yaml_relative_path = r"input\yamls\2025_06_20_config.yaml"
config = load_config(yaml_relative_path=config_yaml_relative_path)

# Load physical model
physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
physical_model = load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)


def get_yamls_and_json(timestamp: str):  # TODO
    pass


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
    json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-21\2025-06-21_T_18-55-17\2025-06-21_T_18-55-17_statistics_logs.json"

    # ----------------------- Eating creatures work -------------------------------
    # Name: 2025-06-22_T_00-15-50
    # Num iterations: same as above (200 frames, 20500 steps)
    # Code changes: fixed problem that carnivores didn't eat at all
    # Result: All died pretty fast (after 7000 steps)
    # Found bug: all creatures could "eat" grass & creature (without gaining energy)
    #                because didn't check digest_dict[food_type] == 0 in eat_food function before eat function
    json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_00-15-50\2025-06-22_T_00-15-50_statistics_logs.json"

    # Name: 2025-06-22_T_00-57-58
    # Num iterations: same as above (200 frames, 20500 steps)
    # Code changes: fixed the bug found above
    # Config changes: REPRODUCTION_ENERGY = 3000 (was 4000)
    json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_00-57-58\2025-06-22_T_00-57-58_statistics_logs.json"

    # Name: 2025-06-22_T_01-33-52
    # Num iterations: much longer run (200 frames, 102500 steps)
    # Result: all died around step 25000
    json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_01-33-52\2025-06-22_T_01-33-52_statistics_logs.json"

    # Name: 2025-06-22_T_12-00-30
    # Num iterations: shorter run to check for carnivores reproduction (15 frames, 10000 steps)
    # Code changes: carnivore that eat creature now get its energy also
    # Result: There are carnivore children! but they didn't survive until the end
    # json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_12-00-30\2025-06-22_T_12-00-30_statistics_logs.json"

    # Name: 2025-06-22_T_12-35-39
    # Num iterations: longer run to check for carnivores children stability (200 frames, 102500 steps)
    # Config changes: back to REPRODUCTION ENERGY = 5000 (was 3000)
    # Result: all died around step 12000
    # json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_12-35-39\2025-06-22_T_12-35-39_statistics_logs.json"

    # Name: 2025-06-22_T_14-22-27
    # Goal: more carnivores at the beginning to achieve carnivores survival
    # Num iterations: 75 frames, 40000 steps
    # Code changes: added num frames and num steps to statistics_logs
    # Config changes: CHANCE_TO_HERBIVORE = 0.3 (was 0.5), less iterations (not needed right now)
    # Result: all died around step 20000
    json_path = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-22\2025-06-22_T_14-22-27\2025-06-22_T_14-22-27_statistics_logs.json"

    # Load and reconstruct statistics logs
    from statistics_logs import StatisticsLogs

    statistics_logs = StatisticsLogs.from_json(filepath=json_path)

    # Plot things
    # statistics_logs.plot_creatures_lifespan_vs_step_matrix()
    # statistics_logs.plot_creatures_nutrition_vs_step_matrix()
    # statistics_logs.plot_creatures_lifespan_graphs()
    statistics_logs.plot_creatures_nutrition_graphs(to_show=True)
    # statistics_logs.plot_num_herbivores_vs_num_carnivores_per_step(to_show=True)
