# main.py
import shutil
import time

from input.codes import config, physical_model
from profiles.profiler import profileit

# Load config
config_yaml_relative_path = r"input\yamls\2025_06_20_config.yaml"
config = config.load_config(yaml_relative_path=config_yaml_relative_path)

# Load physical model
physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
physical_model = physical_model.load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)

from simulation import Simulation


@profileit(output_dir=config.OUTPUT_FOLDER, timestamp=config.timestamp)
def main():
    # Run simulation
    start_time = time.time()
    sim = Simulation()
    sim.run_and_visualize()

    # Plot and save creature statistics and env statistics summary graphs
    sim.statistics_logs.to_json(filepath=config.STATISTICS_LOGS_JSON_FILEPATH)
    sim.statistics_logs.plot_and_save_statistics_graphs(to_save=True)

    # copy config and physical model to output folder
    shutil.copyfile(src=config.yaml_path,
                    dst=config.OUTPUT_FOLDER.joinpath(f"{config.timestamp}_config.yaml"))
    shutil.copyfile(src=physical_model.yaml_path,
                    dst=config.OUTPUT_FOLDER.joinpath(f"{config.timestamp}_physical_model.yaml"))

    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
