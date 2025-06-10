import json
from input.codes import config, physical_model

# Load config
config_yaml_relative_path = r"input\yamls\2025_04_18_config.yaml"
config = config.load_config(yaml_relative_path=config_yaml_relative_path)

# Load physical model
physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
physical_model = physical_model.load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)

from statistics_logs import StatisticsLogs

b = StatisticsLogs.from_json(
    filepath=r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\2025-06-10\2025-06-10_T_18-22-02_statistics_logs.json")

breakpoint()
