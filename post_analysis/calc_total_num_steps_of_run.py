def calc_total_num_steps(up_to_frame: int) -> int:
    total_steps = 0

    keys = list(NUM_STEPS_FROM_FRAME_DICT.keys())
    values = list(NUM_STEPS_FROM_FRAME_DICT.values())

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


if __name__ == '__main__':
    NUM_FRAMES = 15  # the actual number of steps = NUM_FRAMES * UPDATE_ANIMATION_INTERVAL

    # key is frame number and value is num steps per frame from this frame onward
    NUM_STEPS_FROM_FRAME_DICT = {0: 1000,
                                5: 500,
                                }

    print(calc_total_num_steps(up_to_frame=NUM_FRAMES))
