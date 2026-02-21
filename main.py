import time

from input.codes.sim_config import config
from input.codes.physical_model_factory import PhysicalModelFactory
from profiles.profiler import profileit
from simulation import Simulation


@profileit()
def run_sim():
    sim = Simulation()
    sim.run_and_visualize()


if __name__ == "__main__":
    # *** Choose config name in input\codes\config ***

    # Search for available physical models
    PhysicalModelFactory.discover_models()

    # Run simulation multiple times
    for i in range(config.NUM_RUNS):
        start_time = time.time()
        config.update_config()
        run_sim()
        total_time = time.time() - start_time
        print('--------------------------------')
        print(f"\nSimulation {i}/{config.NUM_RUNS} running time: {total_time:.2f} seconds")
        print('--------------------------------')

