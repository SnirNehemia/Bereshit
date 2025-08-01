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

    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
