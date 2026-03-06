import random
import shutil

import numpy as np
from scipy.spatial import KDTree

from b_basic.creatures.creature import Creature
from b_basic.environments.environment import Environment
from b_basic.sim_config import sim_config


def init_environment():
    # Create the environment. Ensure that 'map.png' exists and follows the color conventions.
    return Environment(map_filename=sim_config.config.ENV_PATH,
                       grass_generation_rate=sim_config.config.GRASS_GENERATION_RATE,
                       leaves_generation_rate=sim_config.config.LEAVES_GENERATION_RATE)


def init_creatures(env: Environment, brain_obj) -> dict[int, Creature]:
    """
    Initializes creatures ensuring they are not placed in a forbidden (black) area.
    """
    creatures = dict()
    num_creatures = sim_config.config.NUM_CREATURES
    output_size = sim_config.config.OUTPUT_SIZE

    for creature_id in range(num_creatures):
        # get a valid position
        position = get_valid_position(env=env)

        # static traits
        gen = 0
        parent_id = None
        birth_step = 0
        color = np.random.rand(3)  # Random RGB color.
        max_age = int(np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) *
                      sim_config.config.INIT_MAX_AGE)

        max_mass = np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) * \
                   sim_config.config.INIT_MAX_MASS
        max_height = np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) * \
                     sim_config.config.INIT_MAX_HEIGHT
        max_strength = np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) * \
                       sim_config.config.INIT_MAX_STRENGTH

        max_speed = np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) * \
                    sim_config.config.INIT_MAX_SPEED
        max_energy = np.random.uniform(low=sim_config.config.INIT_MAX_FRACTION, high=1) * \
                     sim_config.config.INIT_MAX_ENERGY

        reproduction_cooldown = sim_config.config.REPRODUCTION_COOLDOWN
        reproduction_energy = sim_config.config.REPRODUCTION_ENERGY

        # choose randomly if creature is herbivore or carnivore
        digest_roll = np.random.rand()
        if digest_roll <= sim_config.config.CHANCE_TO_HERBIVORE:
            digest_dict = sim_config.config.INIT_HERBIVORE_DIGEST_DICT
        else:
            digest_dict = sim_config.config.INIT_CARNIVORE_DIGEST_DICT

        eyes = sim_config.config.EYES
        vision_limit = sim_config.config.VISION_LIMIT
        eyes_dofs = 3 * len(eyes)  # 3 (flag, distance, angle) X num eyes X num channels
        other_dofs = len(sim_config.config.NORM_INPUT) - eyes_dofs
        input_size = other_dofs + eyes_dofs
        brain = brain_obj([input_size, output_size])

        # init creature
        creature = Creature(
            creature_id=creature_id, gen=gen, parent_id=parent_id, birth_step=birth_step, color=color,
            max_age=max_age, max_mass=max_mass, max_height=max_height,
            max_strength=max_strength, max_speed=max_speed, max_energy=max_energy,
            digest_dict=digest_dict,
            reproduction_cooldown=reproduction_cooldown, reproduction_energy=reproduction_energy,
            eyes=eyes, vision_limit=vision_limit, brain=brain,
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


def build_creatures_kd_tree(positions: np.ndarray) -> KDTree:
    """
    Builds a KDTree from the positions of all creatures.
    """
    if len(positions) > 0:
        return KDTree(positions)
    else:
        return KDTree([[0, 0]])


def get_brain_input(creature: Creature, seek_result: dict):
    eyes_inputs = [prepare_eye_input(seek_result, creature.vision_limit)
                   for seek_result in seek_result.values()]
    brain_input = [
        creature.speed,
        creature.energy,
        np.concatenate(eyes_inputs)
    ]
    brain_input = np.hstack(brain_input)
    return brain_input


def prepare_eye_input(seek_result, vision_limit):
    """
    Converts a seek result (distance, signed_angle, idx) or None into a 3-element vector:
      [detection_flag, distance, angle].
    """
    if seek_result is None:
        return np.array([0, vision_limit, 0])
    else:
        distance, angle = seek_result[0:2]
        return np.array([1, distance, angle])


def detect_target_from_kdtree(creature: Creature, i_creature: int,
                              eye_idx: int,
                              kd_tree: KDTree,
                              candidate_points: np.ndarray,
                              candidates_indices_to_remove: list):
    """
    Generic function to detect the closest target from candidate_points using a KDTree.

    Parameters:
      creature: the creature performing the detection.
      eye_idx: creature.eyes[eye_idx] is (angle_offset,aperture) specifying the eye's viewing direction
               relative to the creature's heading and the aperture
      kd_tree: a KDTree built from candidate_points.
      candidate_points: numpy array of shape (N, 2) containing candidate target positions.
      noise_std: standard deviation for optional Gaussian noise.

    Returns:
      A tuple (distance, signed_angle, idx) for the detected target, or None if no target qualifies.
    """
    # Query the KDTree for candidate indices within the creature's vision range.
    eye_position = creature.position
    candidates_indices = kd_tree.query_ball_point(x=eye_position,
                                                  r=creature.vision_limit)

    # Remove self if searching for creatures
    if i_creature != -1:
        if i_creature in candidates_indices:
            candidates_indices.remove(i_creature)

    # Remove irrelevant candidates
    for candidiate_idx_to_remove in candidates_indices_to_remove:
        if candidiate_idx_to_remove in candidates_indices:
            candidates_indices.remove(candidiate_idx_to_remove)

    # Check if empty
    if not candidates_indices:
        return None

    # Get the actual coordinate data for these candidates
    candidates_positions = candidate_points[candidates_indices]

    final_indices, final_distances, final_angles = \
        calc_batch_distance_and_angle(creature=creature,
                                      eye_idx=eye_idx,
                                      eye_position=eye_position,
                                      candidates_positions=candidates_positions,
                                      candidates_indices=candidates_indices)

    # Get closest candidate
    if len(final_distances) > 0:
        closest_candidate_idx = np.argmin(final_distances)
        final_distance = final_distances[closest_candidate_idx]
        final_angle = final_indices[closest_candidate_idx]
        final_idx = final_indices[closest_candidate_idx]
        return final_distance, final_angle, final_idx
    else:
        return None


def get_eye_direction(creature, eye_idx):
    # Compute the eye's viewing direction by rotating the heading by angle_offset.
    heading = creature.get_heading()
    eye_direction = np.array([
        heading[0] * creature.eye_cos_offset[eye_idx] - heading[1] * creature.eye_sin_offset[eye_idx],
        heading[0] * creature.eye_sin_offset[eye_idx] + heading[1] * creature.eye_cos_offset[eye_idx]
    ])

    return eye_direction


def calc_batch_distance_and_angle(creature, eye_idx, eye_position,
                                  candidates_positions,
                                  candidates_indices):
    """
    Optimized batch processing for vision checks.
    returns relative idx, distance, angle
    candidates: (N, 2) array of target positions

    Inside the function:
    1. Start with all query_indices: [5, 12, 42]
    2. Apply Distance Mask: [True, False, True] -> results in [5, 42]
    3. Apply FOV Mask: [True, False] -> results in [5]
    4. Return value: [5] (The global index of the visible creature)
    """
    # 1. Vectorized distance calculation
    target_vectors = candidates_positions - eye_position  # target_vectors: (N, 2)
    dist_sq = np.sum(target_vectors ** 2, axis=1)  # dist_sq: (N,)

    # 2. Distance Mask (Filter out self and out-of-range)
    dist_mask = (dist_sq > 1e-9) & (dist_sq <= creature.vision_limit_sq)

    if not np.any(dist_mask):
        return [], [], []

    # Apply mask and get actual distances
    valid_targets = target_vectors[dist_mask]
    valid_distances = np.sqrt(dist_sq[dist_mask])
    valid_indices = np.array(candidates_indices)[dist_mask]

    # Dot product for FOV (the "Broad-to-Narrow" filter) (N_valid,)
    eye_direction = get_eye_direction(creature=creature, eye_idx=eye_idx)
    dots = (valid_targets[:, 0] * eye_direction[0] + valid_targets[:, 1] * eye_direction[
        1]) / valid_distances  # det: (N_valid,)
    fov_mask = dots >= creature.eye_cos_half_aperture[eye_idx]
    if not np.any(fov_mask): return [], [], []

    final_targets = valid_targets[fov_mask]
    final_distances = valid_distances[fov_mask]
    final_dots = dots[fov_mask]
    final_indices = valid_indices[fov_mask]

    # Cross product (determinant) for the angle sign
    dets = eye_direction[0] * final_targets[:, 1] - eye_direction[1] * final_targets[:, 0]
    final_angles = np.arctan2(dets, final_dots * final_distances)

    # 6. Add Noise to final results
    noise_std = sim_config.config.NOISE_STD
    if noise_std > 0:
        final_distances += np.random.normal(0, noise_std, size=final_distances.shape)
        final_angles += np.random.normal(0, noise_std, size=final_angles.shape)

    return final_indices, final_distances, final_angles


def do_purge(num_creatures_threshold: int,
             creatures: dict[int, Creature],
             dead_creatures: dict[int, Creature],
             step_counter: int,
             statistics_logs):
    """
    Do purge if its time (PURGE_STEP_FREQUENCY passed) or there are too many creatures
    :param num_creatures_threshold:
    :param creatures:
    :param dead_creatures:
    :param step_counter:
    :param statistics_logs:
    :return:
    """
    purged_creatures_ids = []
    is_time_to_purge = step_counter % sim_config.config.PURGE_STEP_FREQUENCY == 0
    num_creatures_to_purge = len(creatures) - num_creatures_threshold
    if is_time_to_purge or num_creatures_to_purge > 0:
        # kill all always-slow creatures
        too_slow_creatures_ids = []
        for creature_id, creature in creatures.items():
            if creature.max_speed_exp <= sim_config.config.PURGE_SPEED_THRESHOLD:
                too_slow_creatures_ids.append(creature_id)

        kill_creatures(creatures_ids_to_kill=too_slow_creatures_ids,
                       creatures=creatures, dead_creatures=dead_creatures)
        purged_creatures_ids.extend(too_slow_creatures_ids)

        # Kill more randomly if needed
        num_creatures_to_purge = len(creatures) - num_creatures_threshold
        if num_creatures_to_purge > 0:
            creature_ids = list(creatures.keys())
            random_creatures_ids = random.sample(creature_ids, num_creatures_to_purge)

            kill_creatures(creatures_ids_to_kill=random_creatures_ids,
                           creatures=creatures, dead_creatures=dead_creatures)
            purged_creatures_ids.extend(random_creatures_ids)

    # Add tp statistics logs death causes dict
    statistics_logs.death_causes_dict['purge'].extend(purged_creatures_ids)

    return purged_creatures_ids


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
                        num_children: int,
                        step_counter: int):
    new_child_ids = []
    for creature_id in creatures_ids_to_reproduce:
        # update parent
        parent = creatures[creature_id]
        child = parent.reproduce()
        parent.log.add_record('reproduce', step_counter)

        # update child
        id_count += 1
        child.creature_id = id_count
        child.birth_step = step_counter
        child.log.creature_id = child.creature_id

        # add child to simulation
        creatures[id_count] = child
        new_child_ids.append(id_count)
        num_children += 1

    return new_child_ids, num_children, id_count


def update_creatures_logs(creatures: dict[int, Creature]):
    for creature in creatures.values():
        creature.log.add_record('energy', creature.energy)
        creature.log.add_record('speed', creature.speed)


def update_kd_trees(env: Environment,
                    positions: np.ndarray,
                    creatures_kd_tree: KDTree,
                    to_update_kd_tree: dict[bool],
                    step_counter: int):
    # Update KDTree if needed or every "kdtree_update_interval" steps
    is_time_to_update_kd_trees = step_counter % sim_config.config.UPDATE_KDTREE_INTERVAL == 0
    if to_update_kd_tree['grass'] or is_time_to_update_kd_trees:
        env.update_grass_kd_tree()

    if to_update_kd_tree['leaf'] or is_time_to_update_kd_trees:
        pass

    if to_update_kd_tree['creature'] or is_time_to_update_kd_trees:
        creatures_kd_tree = build_creatures_kd_tree(positions=positions)

    return creatures_kd_tree


def calc_num_steps_per_frame(frame: int) -> int:
    keys = list(sim_config.config.NUM_STEPS_FROM_FRAME_DICT.keys())
    previous_value = sim_config.config.NUM_STEPS_FROM_FRAME_DICT[keys[0]]

    for key, value in sim_config.config.NUM_STEPS_FROM_FRAME_DICT.items():
        if frame < key:
            break
        else:
            previous_value = value

    num_steps_per_frame = previous_value

    return num_steps_per_frame


def calc_total_num_steps(num_steps_from_frame_dict: dict, up_to_frame: int) -> int:
    total_num_steps = 0

    up_to_frame_list = list(num_steps_from_frame_dict.keys())
    num_steps_up_to_frame_list = list(num_steps_from_frame_dict.values())

    for i in range(len(up_to_frame_list)):
        start = up_to_frame_list[i]

        # Determine end of this interval
        if i + 1 < len(up_to_frame_list):
            end = min(up_to_frame_list[i + 1], up_to_frame)
        else:
            end = up_to_frame

        if start >= up_to_frame:
            break  # no need to continue

        total_num_steps += (end - start) * num_steps_up_to_frame_list[i]

    return total_num_steps


def check_abort_simulation(creatures: dict[int, Creature], step_counter: int):
    abort_simulation = False
    if len(creatures) > sim_config.config.MAX_NUM_CREATURES:
        print(f'step={step_counter}: Too many creatures, simulation is too slow.')
        abort_simulation = True
    elif len(creatures) <= 0:
        print(f'\nstep={step_counter}: all creatures are dead :(.')
        abort_simulation = True

    return abort_simulation


def copy_config_file_to_output_folder():
    shutil.copyfile(src=sim_config.config.full_path,
                    dst=sim_config.config.OUTPUT_FOLDER.joinpath(f"{sim_config.config.timestamp}_config.yaml"))


if __name__ == '__main__':

    # Example
    # num_frames = 10
    # num_steps_from_frame_dict = {0: 1000,
    #                              5: 500,
    #                              }

    # Load from config
    config_name = "2026_02_24_config_pm1.yaml"
    sim_config.load_config(config_name=config_name)
    num_frames = sim_config.config.NUM_FRAMES
    num_steps_from_frame_dict = sim_config.config.NUM_STEPS_FROM_FRAME_DICT

    # Calc num steps up to given frame
    for frame in range(int(0.8 * num_frames), num_frames):
        num_steps = calc_total_num_steps(
            num_steps_from_frame_dict=num_steps_from_frame_dict,
            up_to_frame=frame)
        print(f'up to {frame=}: {num_steps=}')
