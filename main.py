# main.py
import time
from simulation import Simulation

if __name__ == "__main__":
    start_time = time.time()
    sim = Simulation()
    sim.run_and_visualize()
    total_time = time.time() - start_time
    print(f"Total simulation time: {total_time:.2f} seconds")
