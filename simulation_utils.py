import importlib

import numpy as np
from scipy.spatial import KDTree

from config import Config as config
from creature import Creature
from environment import Environment

brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
Brain = getattr(brain_module, 'Brain')


def initialize_creatures(num_creatures, simulation_space, input_size, output_size,
                         eyes_params, env: Environment) -> dict[int, Creature]:
    """
    Initializes creatures ensuring they are not placed in a forbidden (black) area.
    """
    creatures = dict()

    for creature_id in range(num_creatures):
        # get a valid position.
        position = []
        valid_position = False
        while not valid_position:
            position = np.random.rand(2) * simulation_space
            # Convert (x, y) to indices (col, row)
            col, row = int(position[0]), int(position[1])
            # height, width = env.map_data.shape[:2]
            # Check bounds and obstacle mask.
            if col < 0 or col >= env.width or row < 0 or row >= env.height:
                continue
            if env.obstacle_mask[row, col]:
                continue
            valid_position = True

        # static traits
        gen = 0
        parent_id = None
        birth_step = 0
        max_age = np.random.randint(low=config.INIT_MAX_AGE * 0.8, high=config.INIT_MAX_AGE)
        color = np.random.rand(3)  # Random RGB color.

        max_mass = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_MASS
        max_height = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_HEIGHT
        max_strength = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_STRENGTH

        max_speed = np.random.uniform(low=0.8, high=1) * config.MAX_SPEED
        max_energy = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_ENERGY

        digest_dict = config.INIT_DIGEST_DICT
        reproduction_energy = config.REPRODUCTION_ENERGY

        vision_limit = config.VISION_LIMIT
        brain = Brain([input_size, output_size])

        # init creature
        creature = Creature(
            creature_id=creature_id, gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
            max_age=max_age, max_mass=max_mass, max_height=max_height,
            max_strength=max_strength, max_speed=max_speed, max_energy=max_energy,
            digest_dict=digest_dict, reproduction_energy=reproduction_energy,
            eyes_params=eyes_params, vision_limit=vision_limit, brain=brain,
            position=position)

        creatures[creature_id] = creature
    return creatures


def build_creatures_kd_tree(creatures: dict[int, Creature]) -> KDTree:
    """
    Builds a KDTree from the positions of all creatures.
    """
    positions = [creature.position for creature in creatures.values()]
    if positions:
        return KDTree(positions)
    else:
        return KDTree([[0, 0]])


def update_creatures_kd_tree(creatures: dict[int, Creature]) -> KDTree:
    return build_creatures_kd_tree(creatures)


def seek(creatures: dict[int, Creature], creatures_kd_tree: KDTree, env: Environment,
         creature: Creature, noise_std: float = 0.0):
    """
    Uses the specified eye (given by eye_params: (angle_offset, aperture))
    to detect a nearby target.
    Computes the eye's viewing direction by rotating the creature's heading by angle_offset.
    Returns (distance, signed_angle) if a target is found within half the aperture, else None.
    """
    channel_results = {}

    channels_list = []
    kd_tree = []
    for i_eye, eye_params in enumerate(creature.eyes_params):
        for channel in config.EYE_CHANNEL:
            candidate_points = np.array([])
            if channel == 'grass':
                if len(env.grass_points) > 0:
                    kd_tree = env.grass_kd_tree
                    candidate_points = np.array(env.grass_points)
            # elif channel == 'leaves':
            #     if len(env.leaf_points) > 0:
            #         candidate_points = np.array(env.leaf_points)
            # elif channel == 'water':
            #     candidate_points = np.array([[env.water_source[0], env.water_source[1]]])
            elif channel == 'creatures':
                kd_tree = creatures_kd_tree
                candidate_points = np.array([c.position for c in creatures.values()])

            if len(candidate_points) > 0:
                # kd_tree = KDTree(candidate_points)
                result = detect_target_from_kdtree(creature, eye_params, kd_tree, candidate_points, noise_std)
            else:
                result = None

            channel_name = f'{channel}_{i_eye}'
            channel_results[channel_name] = result
            channels_list.append(channel_name)

    return channel_results


def use_brain(creature: Creature, env: Environment, seek_results: dict, dt: float):
    try:
        # get brain input
        eyes_inputs = [prepare_eye_input(seek_result, creature.vision_limit) for seek_result in
                       seek_results.values()]
        brain_input = []
        brain_input.append(np.array([creature.hunger, creature.thirst]))
        brain_input.append(creature.speed)
        brain_input.append(np.concatenate(eyes_inputs))
        brain_input = np.hstack(brain_input)
        decision = creature.think(brain_input)
        creature.move(decision=decision, dt=dt)
    except Exception as e:
        print(f'Error in Simulation (use_brain, movement) for creature: {creature.creature_id}:\n{e}')
        # breakpoint()

    # Collision detection: handle cases where creature's new position is inside an obstacle or outbound.
    try:
        col, row = map(int, creature.position)  # Convert (x, y) to image indices (col, row)
        height, width = env.map_data.shape[:2]
        if col < 0 or col >= width or row < 0 or row >= height or env.obstacle_mask[row, col]:
            # choose if the velocity is set to zero or get mirrored
            if config.BOUNDARY_CONDITION == 'zero':
                creature.velocity = np.array([0.0, 0.0])
            elif config.BOUNDARY_CONDITION == 'mirror':
                creature.velocity = -creature.velocity
    except Exception as e:
        print(f'exception in use_brain for creature: {creature.creature_id}\n{e}')
        print(f'Error in Simulation (use_brain, collision detection) for creature: {creature.creature_id}:\n{e}')
        # breakpoint()


def prepare_eye_input(detection_result, vision_limit):
    """
    Converts a detection result (distance, signed_angle) or None into a 3-element vector:
      [detection_flag, distance, angle].
    """
    if detection_result is None:
        return np.array([0, vision_limit, 0])
    else:
        distance, angle = detection_result[0:2]
        return np.array([1, distance, angle])


def detect_target_from_kdtree(creature: Creature, eye_params,
                              kd_tree: KDTree, candidate_points: np.ndarray,
                              noise_std: float = 0.0):
    """
    Generic function to detect the closest target from candidate_points using a KDTree.

    Parameters:
      creature: the creature performing the detection.
      eye_params: (angle_offset,aperture) specifying the eye's viewing direction relative to the creature's heading.
      kd_tree: a KDTree built from candidate_points.
      candidate_points: numpy array of shape (N, 2) containing candidate target positions.
      noise_std: standard deviation for optional Gaussian noise.

    Returns:
      A tuple (distance, signed_angle) for the detected target, or None if no target qualifies.
    """
    eye_position = creature.position
    heading = creature.get_heading()
    angle_offset, aperture = eye_params
    # Compute the eye's viewing direction by rotating the heading by angle_offset.
    cos_offset = np.cos(angle_offset)
    sin_offset = np.sin(angle_offset)
    eye_direction = np.array([
        heading[0] * cos_offset - heading[1] * sin_offset,
        heading[0] * sin_offset + heading[1] * cos_offset
    ])
    # Query the KDTree for candidate indices within the creature's vision range.
    candidate_indices = kd_tree.query_ball_point(eye_position, creature.vision_limit)
    best_distance = float('inf')
    detected_info = None
    # Evaluate each candidate - which was sorted by the KDtree.
    for idx in candidate_indices:
        candidate = candidate_points[idx]
        # Skip if the candidate is the creature itself.
        if np.allclose(candidate, creature.position):
            continue
        target_vector = candidate - eye_position
        distance = np.linalg.norm(target_vector)
        if distance == 0 or distance > creature.vision_limit:
            continue
        target_direction = target_vector / distance
        dot = np.dot(eye_direction, target_direction)
        det = eye_direction[0] * target_direction[1] - eye_direction[1] * target_direction[0]
        angle = np.arctan2(det, dot)
        # Only accept targets within half the aperture.
        if abs(angle) > (aperture / 2):
            continue
        if noise_std > 0:
            distance += np.random.normal(0, noise_std)
            angle += np.random.normal(0, noise_std)
        if distance < best_distance:
            best_distance = distance
            detected_info = (distance, angle, idx)  # TODO: make sure to save the index too for fast removal
    return detected_info


def eat_food(creature: Creature, env: Environment,
             seek_result: dict, food_type: str, step_counter: int):
    # check if creature is full
    if creature.energy >= creature.max_energy:
        return False

    # get food points
    is_eat = False
    food_points, food_energy = [], 0

    # This for serves the case of several eyes
    for key, value in seek_result.items():
        if key.startswith(food_type) and not value == None:
            food_points.append(value)

    if food_type == 'grass':
        food_energy = config.GRASS_ENERGY
    elif food_type == 'leaf':
        food_energy = config.LEAF_ENERGY

    if len(food_points) > 0:
        if len(food_points) > 1:
            # candidate_indices = kd_tree.query_ball_point(eye_position, creature.vision_limit) TODO: use the kd_tree here too!
            food_distances = [food_point[:2] for food_point in food_points]
            # DELETE: food_distances = [np.linalg.norm(food_point[:2] - creature.position)
            #                   for food_point in food_points]
            closest_food_index = np.argmin(food_distances)
        else:
            food_distances = food_points[0][:2]
            closest_food_index = 0
        closest_food_distance = np.min(food_distances[closest_food_index])
        closest_food_point = env.grass_points[food_points[closest_food_index][2]]
        # if someone got there first
        if closest_food_point in env.grass_remove_list:
            return False

        if closest_food_distance <= config.FOOD_DISTANCE_THRESHOLD:
            # creature eat food
            creature.eat(food_type=food_type, food_energy=food_energy)
            creature.log_eat.append(step_counter)

            # remove food from environment
            env.grass_remove_list.append(closest_food_point)
            # self.env.grass_points.remove(closest_food_point)  # TODO:moved outside - check if it's fine
            # self.env.update_grass_kd_tree()  # TODO:moved outside - check if it's fine
            is_eat = True

    return is_eat


def calc_num_steps_per_frame(frame: int) -> int:
    keys = list(config.NUM_STEPS_FROM_FRAME_DICT.keys())
    previous_value = config.NUM_STEPS_FROM_FRAME_DICT[keys[0]]

    for key, value in config.NUM_STEPS_FROM_FRAME_DICT.items():
        if frame < key:
            break
        else:
            previous_value = value

    num_steps_per_frame = previous_value

    return num_steps_per_frame


def calc_total_num_steps(up_to_frame: int) -> int:
    total_steps = 0

    keys = list(config.NUM_STEPS_FROM_FRAME_DICT.keys())
    values = list(config.NUM_STEPS_FROM_FRAME_DICT.values())

    for i in range(len(keys)):
        start = keys[i]

        # Determine end of this interval
        if i + 1 < len(keys):
            end = min(keys[i + 1], up_to_frame)
        else:
            end = up_to_frame

        if start >= up_to_frame:
            break  # no need to continue

        total_steps += (end - start) * values[i]

    return total_steps


def check_abort_simulation(creatures: dict[int, Creature], step_counter: int):
    abort_simulation = False
    if len(creatures) > config.MAX_NUM_CREATURES:
        print(f'step={step_counter}: Too many creatures, simulation is too slow.')
        abort_simulation = True
    elif len(creatures) <= 0:
        print(f'\nstep={step_counter}: all creatures are dead :(.')
        abort_simulation = True

    return abort_simulation


if __name__ == '__main__':
    for frame in range(10):
        num_steps = calc_num_steps_per_frame(frame=frame)
        print(f'{frame=}: {num_steps=}')
