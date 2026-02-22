import cProfile
import functools
import os

from input.codes import sim_config


def profileit():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # make sure directory exists
            os.makedirs(sim_config.config.OUTPUT_FOLDER, exist_ok=True)

            # get filename with timestamp
            filepath = f"{sim_config.config.OUTPUT_FOLDER}/{sim_config.config.timestamp}_{func.__name__}_profiler.prof"
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                profiler.dump_stats(filepath)
                # print(f"Profiler name: {Path(filepath).stem}")
            return result

        return wrapper

    return decorator
