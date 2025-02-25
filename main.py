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
            position = np.random.rand(2) * simulation_space
            # Convert (x, y) to indices (col, row)
            col, row = int(position[0]), int(position[1])
            height, width = env.map_data.shape[:2]
            # Check bounds and obstacle mask.
            if col < 0 or col >= width or row < 0 or row >= height:
                continue
            if env.obstacle_mask[row, col]:
                continue
            valid_position = True

        # static traits
        max_age = 50
        max_weight = 10.0
        max_height = 5.0
        max_speed = [5.0, 5.0]
        color = np.random.rand(3)  # Random RGB color.

        energy_efficiency = 1
        speed_efficiency = 0.1
        food_efficiency = 1
        reproduction_energy = 80

        vision_limit = 100.0
        brain = Brain([input_size, output_size])

        # dynamic traits
        weight = np.random.rand() * max_weight
        height = np.random.rand() * max_height
        speed = (np.random.rand(2) - 0.5) * max_speed
        energy = np.random.rand() * 1000
        hunger = np.random.rand() * 10
        thirst = np.random.rand() * 10

        # init creature
        creature = Creature(max_age=max_age, max_weight=max_weight, max_height=max_height, max_speed=max_speed,
                            color=color,
                            energy_efficiency=energy_efficiency, speed_efficiency=speed_efficiency,
                            food_efficiency=food_efficiency, reproduction_energy=reproduction_energy,
                            eyes_params=eyes_params, vision_limit=vision_limit, brain=brain,
                            weight=weight, height=height,
                            position=position, speed=speed, energy=energy, hunger=hunger, thirst=thirst)

        creatures.append(creature)
    return creatures


if __name__ == "__main__":
    start_time = time.time()

    # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
    env = Environment(map_filename=config.ENV_PATH,
                      grass_generation_rate=config.GRASS_GENERATION_RATE,
                      leaves_generation_rate=config.LEAVES_GENERATION_RATE)

    # Initialize creatures (ensuring they are not in forbidden areas).
    creatures = initialize_creatures(num_creatures=config.NUM_CREATURES,
                                     simulation_space=config.SIMULATION_SPACE,
                                     input_size=config.INPUT_SIZE,
                                     output_size=config.OUTPUT_SIZE,
                                     eyes_params=config.EYES_PARAMS,
                                     env=env)

    sim = Simulation(creatures, env)

    sim.run_and_visualize(dt=config.DT,
                          noise_std=config.NOISE_STD,
                          frames=config.NUM_FRAMES,
                          save_filename=config.SAVE_FILENAME)

    total_time = time.time() - start_time
    print("Simulation animation saved as simulation.mp4")
    print(f"Total simulation time: {total_time:.2f} seconds")
