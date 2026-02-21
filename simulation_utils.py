import shutil

import numpy as np
from scipy.spatial import KDTree

from input.codes.sim_config import config
from creature import Creature
from environment import Environment


def init_environment():
    # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
    return Environment(map_filename=config.ENV_PATH,
                       grass_generation_rate=config.GRASS_GENERATION_RATE,
                       leaves_generation_rate=config.LEAVES_GENERATION_RATE)


def init_creatures(env: Environment, brain_obj) -> dict[int, Creature]:
    """
    Initializes creatures ensuring they are not placed in a forbidden (black) area.
    """
    creatures = dict()
    num_creatures = config.NUM_CREATURES
    output_size = config.OUTPUT_SIZE

    for creature_id in range(num_creatures):
        # get a valid position
        position = get_valid_position(env=env)

        # static traits
        gen = 0
        parent_id = None
        birth_step = 0
        color = np.random.rand(3)  # Random RGB color.
        max_age = int(np.random.uniform(low=0.8, high=1) * config.INIT_MAX_AGE)

        max_mass = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_MASS
        max_height = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_HEIGHT
        max_strength = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_STRENGTH

        max_speed = np.random.uniform(low=0.8, high=1) * config.MAX_SPEED
        max_energy = np.random.uniform(low=0.8, high=1) * config.INIT_MAX_ENERGY

        reproduction_energy = config.REPRODUCTION_ENERGY

        # choose randomly if creature is herbivore or carnivore
        digest_roll = np.random.rand()
        if digest_roll <= config.CHANCE_TO_HERBIVORE:
            digest_dict = config.INIT_HERBIVORE_DIGEST_DICT
            eyes_channels = config.EYE_CHANNEL  # sees only grass: [config.EYE_CHANNEL[0]]
            eyes_params = config.EYES_PARAMS  # sees only grass: config.EYES_PARAMS[0]
        else:
            digest_dict = config.INIT_CARNIVORE_DIGEST_DICT
            eyes_channels = config.EYE_CHANNEL  # sees only creatures: [config.EYE_CHANNEL[1]]
            eyes_params = config.EYES_PARAMS  # sees only creatures: [config.EYES_PARAMS[1]]

        vision_limit = config.VISION_LIMIT
        eyes_dofs = 3 * len(eyes_params) * len(eyes_channels)  # 3 (flag, distance, angle) for each eye * X channels
        other_dofs = 3  # hunger, thirst, speed
        input_size = other_dofs + eyes_dofs
        brain = brain_obj([input_size, output_size])

        # init creature
        creature = Creature(
            creature_id=creature_id, gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
            max_age=max_age, max_mass=max_mass, max_height=max_height,
            max_strength=max_strength, max_speed=max_speed, max_energy=max_energy,
            digest_dict=digest_dict, reproduction_energy=reproduction_energy,
            eyes_channels=eyes_channels, eyes_params=eyes_params, vision_limit=vision_limit,
            brain=brain,
            position=position)

        creatures[creature_id] = creature
    return creatures


def get_valid_position(env: Environment):
    position = []
    valid_position = False
    while not valid_position:
        position = np.random.rand(2) * env.size
        # Convert (x, y) to indices (col, row)
        col, row = int(position[0]), int(position[1])
        # height, width = env.map_data.shape[:2]
        # Check bounds and obstacle mask.
        if col < 0 or col >= env.width or row < 0 or row >= env.height:
            continue
        if env.obstacle_mask[row, col]:
            continue
        valid_position = True

    return position


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


def detect_collision(creature, env):
    """
    Handle cases where creature's new position is inside an obstacle or outbound.
    :param creature:
    :param env:
    :return:
    """
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
        print(f'Error in Simulation (use_brain, collision detection) for creature: {creature.creature_id}:\n{e}')
        # breakpoint()


def get_brain_input(creature: Creature, seek_result: dict):
    eyes_inputs = [prepare_eye_input(seek_result, creature.vision_limit)
                   for seek_result in seek_result.values()]
    brain_input = [
        np.array([creature.hunger, creature.thirst]),
        creature.speed,
        np.concatenate(eyes_inputs)
    ]
    brain_input = np.hstack(brain_input)
    return brain_input


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
                              kd_tree: KDTree,
                              candidate_points: np.ndarray,
                              candidates_to_remove_list: list,
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
      A tuple (distance, signed_angle, idx) for the detected target, or None if no target qualifies.
    """
    try:
        # get eye position and direction
        eye_position, eye_direction, eye_aperture = \
            get_eye_position_and_direction(creature=creature,
                                           eye_params=eye_params)

        # Query the KDTree for candidate indices within the creature's vision range.
        candidate_indices = kd_tree.query_ball_point(x=eye_position,
                                                     r=creature.vision_limit)

        # Check which candidate indices satisfies the eye aperture and choose the closest one
        best_distance = float('inf')
        detected_info = None
        for idx in candidate_indices:
            try:
                candidate = candidate_points[idx]
            except IndexError:  # TODO - need to fix: maybe candidates_to_remove_list is needed
                continue

            is_relevant, distance, angle = \
                calc_distance_and_angle_of_target(candidate=candidate, creature=creature,
                                                  eye_position=eye_position,
                                                  eye_direction=eye_direction,
                                                  eye_aperture=eye_aperture,
                                                  noise_std=noise_std)

            if not is_relevant:
                continue

            # update detected_info if current target is closer
            if distance < best_distance:
                best_distance = distance
                detected_info = (distance, angle, idx)
    except Exception as e:
        print(e)
        breakpoint()
    return detected_info


def get_eye_position_and_direction(creature, eye_params):
    eye_position = creature.position
    heading = creature.get_heading()
    angle_offset, eye_aperture = eye_params
    # Compute the eye's viewing direction by rotating the heading by angle_offset.
    cos_offset = np.cos(angle_offset)
    sin_offset = np.sin(angle_offset)
    eye_direction = np.array([
        heading[0] * cos_offset - heading[1] * sin_offset,
        heading[0] * sin_offset + heading[1] * cos_offset
    ])

    return eye_position, eye_direction, eye_aperture


def calc_distance_and_angle_of_target(candidate, creature,
                                      eye_position, eye_direction, eye_aperture,
                                      noise_std):
    """
    Check if candidate is relevant given creature eye.
    :param candidate:
    :param creature:
    :param eye_position:
    :param eye_direction:
    :param eye_aperture:
    :return: (is_relevant, distance, angle).
    """
    # Skip if the candidate is the creature itself.
    if np.allclose(candidate, creature.position):
        return False, None, None

    # Check that target is in vision limit # TODO - why is it needed? KDtree check that
    target_vector = candidate - eye_position
    distance = np.linalg.norm(target_vector)
    if distance == 0 or distance > creature.vision_limit:
        return False, None, None

    # Only accept targets within half the aperture
    target_direction = target_vector / distance
    dot = np.dot(eye_direction, target_direction)
    det = eye_direction[0] * target_direction[1] - \
          eye_direction[1] * target_direction[0]
    angle = np.arctan2(det, dot)

    if abs(angle) > (eye_aperture / 2):
        return False, None, None

    # Add noise to seek result
    if noise_std > 0:
        distance += np.random.normal(0, noise_std)
        angle += np.random.normal(0, noise_std)

    return True, distance, angle


def do_purge(do_purge: bool,
             creatures: dict[int, Creature],
             creatures_ids_to_kill: list[int],
             creatures_ids_to_reproduce: list[int],
             step_counter: int):
    creatures_ids_to_purge = []
    if config.DO_PURGE:
        if do_purge:  # criterion met
            purge_count = 0
            for creature_id, creature in creatures.items():
                is_creature_can_be_killed = creature_id not in creatures_ids_to_kill and \
                                            creature_id not in creatures_ids_to_reproduce

                if is_creature_can_be_killed:
                    # kill randomly
                    if np.random.rand(1) < 0.1:
                        purge_count += 1
                        creatures_ids_to_purge.append(creature_id)

                    # kill if creature is always slow
                    if creature.max_speed_exp <= config.PURGE_SPEED_THRESHOLD:
                        purge_count += 1
                        creatures_ids_to_purge.append(creature_id)
            # print(f'\nStep {step_counter}: Purging {purge_count} creatures.')
            do_purge = False

    return do_purge, creatures_ids_to_purge


def kill_creatures(creatures_ids_to_kill: list[int],
                   creatures: dict[int, Creature],
                   dead_creatures: dict[int, Creature]):
    for creature_id in creatures_ids_to_kill:
        dead_creatures[creature_id] = creatures[creature_id]
        del creatures[creature_id]
    return creatures_ids_to_kill


def reporduce_creatures(creatures_ids_to_reproduce: list[int],
                        creatures: dict[int, Creature],
                        id_count: int,
                        children_num: int,
                        step_counter: int):
    new_child_ids = []
    for creature_id in creatures_ids_to_reproduce:
        # update creature
        creature = creatures[creature_id]
        child = creature.reproduce()
        creature.log.add_record('reproduce', step_counter)

        # update child
        id_count += 1
        child.creature_id = id_count
        child.birth_step = step_counter
        child.log.creature_id = child.creature_id

        # add to simulation
        creatures[id_count] = child
        new_child_ids.append(id_count)
        children_num += 1

    return new_child_ids, children_num, id_count


def update_creatures_logs(creatures: dict[int, Creature]):
    for creature in creatures.values():
        creature.log.add_record('energy', creature.energy)
        creature.log.add_record('speed', creature.speed)


def update_environment_and_kd_trees(env: Environment,
                                    creatures: dict[int, Creature],
                                    creatures_kd_tree: KDTree,
                                    to_update_kd_tree: dict[bool],
                                    step_counter: int):
    # Add new grass points to waiting list (will be updated if is
    env.update()

    # Update KDTree if needed or every "kdtree_update_interval" steps
    is_time_to_update_kd_trees = step_counter % config.UPDATE_KDTREE_INTERVAL == 0
    if to_update_kd_tree['grass'] or is_time_to_update_kd_trees:
        env.update_grass_kd_tree()

    if to_update_kd_tree['leaf'] or is_time_to_update_kd_trees:
        pass

    if to_update_kd_tree['creature'] or is_time_to_update_kd_trees:
        creatures_kd_tree = update_creatures_kd_tree(creatures=creatures)

    return creatures_kd_tree


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

def copy_config_and_physical_model_to_output_folder(physical_model_full_path):
    shutil.copyfile(src=config.full_path,
                    dst=config.OUTPUT_FOLDER.joinpath(f"{config.timestamp}_config.yaml"))
    shutil.copyfile(src=physical_model_full_path,
                    dst=config.OUTPUT_FOLDER.joinpath(f"{config.timestamp}_physical_model.yaml"))

if __name__ == '__main__':
    for frame in range(10):
        num_steps = calc_num_steps_per_frame(frame=frame)
        print(f'{frame=}: {num_steps=}')
