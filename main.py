# main.py
import config
import time
import numpy as np
from brain import Brain
from creature import Creature
from simulation import Simulation
from environment import Environment


def initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                         eyes_params, env: Environment):
    """
    Initializes creatures ensuring they are not placed in a forbidden (black) area.
    """
    creatures = []
    for _ in range(num_creatures):
        valid_position = False
        while not valid_position:
            pos = np.random.rand(2) * simulation_space
            # Convert (x, y) to indices (col, row)
            col, row = int(pos[0]), int(pos[1])
            height, width = env.map_data.shape[:2]
            # Check bounds and obstacle mask.
            if col < 0 or col >= width or row < 0 or row >= height:
                continue
            if env.obstacle_mask[row, col]:
                continue
            valid_position = True
        speed = (np.random.rand(2) - 0.5) * 5
        max_size = 10.0
        max_speed = 5.0
        vision_limit = 100.0
        brain = Brain(input_size, output_size)
        hunger = np.random.rand() * 10
        thirst = np.random.rand() * 10
        color = np.random.rand(3)  # Random RGB color.
        creature = Creature(pos, max_size, max_speed,
                            eyes_params,
                            vision_limit, brain, speed, hunger, thirst, color)
        creatures.append(creature)
    return creatures


if __name__ == "__main__":
    start_time = time.time()

    num_creatures = 200
    simulation_space = 1000

    # Define eye parameters: (angle_offset in radians, aperture in radians)
    eyes_params = [(np.radians(30), np.radians(60)),(np.radians(-30), np.radians(60))]

    # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
    env = Environment("Penvs\\Env1.png", grass_generation_rate=3, leaves_generation_rate=1)
    # parameters of network
    input_size = 2 + 2 + 3 * len(
        eyes_params) * 4  # 2 for position, 2 for speed, 3 (flag, distance, angle) for each eye * 4 channels
    output_size = 2
    # Initialize creatures (ensuring they are not in forbidden areas).
    creatures = initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                                     eyes_params, env)

    sim = Simulation(creatures, env)

    noise_std = 0.5
    dt = 1.0
    frames = 75

    sim.run_and_visualize(dt, noise_std, frames, save_filename="simulation.mp4")

    total_time = time.time() - start_time
    print("Simulation animation saved as simulation.mp4")
    print(f"Total simulation time: {total_time:.2f} seconds")
