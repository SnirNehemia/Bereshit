from b_basic.sim_config.codes import sim_config
from c_models.physical_model_factory import PhysicalModelFactory
from d_controllers.simulation import Simulation
from f_post_analysis.profiles.profiler import profileit


@profileit()
def run_sim():
    sim = Simulation()
    return sim.run_and_visualize()


if __name__ == "__main__":
    # Load config
    config_name = "2026_02_24_config_pm1.yaml"
    sim_config.load_config(config_name=config_name)

    # Search for available physical models
    PhysicalModelFactory.discover_models()

    # Run simulation multiple times
    for i in range(sim_config.config.NUM_RUNS):
        sim_config.config.update_config()
        total_time = run_sim()
        print('\n----------------------------------------')
        print(f"Simulation {sim_config.config.timestamp} running time "
              f"({i + 1}/{sim_config.config.NUM_RUNS}): {total_time:.2f} seconds")
        print('----------------------------------------')
