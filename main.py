import time

from input.codes import sim_config
from input.codes.physical_model_factory import PhysicalModelFactory
from profiles.profiler import profileit
from simulation import Simulation


@profileit()
def run_sim():
    sim = Simulation()
    sim.run_and_visualize()


if __name__ == "__main__":
    # Load config
    config_name = "2026_02_19_config.yaml"
    sim_config.load_config(config_name=config_name)

    # Search for available physical models
    PhysicalModelFactory.discover_models()

    # Run simulation multiple times
    for i in range(sim_config.config.NUM_RUNS):
        start_time = time.time()
        sim_config.config.update_config()
        run_sim()
        total_time = time.time() - start_time
        print('\n----------------------------------------')
        print(f"Simulation {sim_config.config.timestamp} running time "
              f"({i}/{sim_config.config.NUM_RUNS}): {total_time:.2f} seconds")
        print('----------------------------------------')
