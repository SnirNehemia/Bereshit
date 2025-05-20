import cProfile
import functools
import os
from datetime import datetime

from input.codes.repos_utils import fetch_directory

OUTPUT_DIR = fetch_directory().joinpath('profiles')


def profileit(output_dir=OUTPUT_DIR):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # make sure directory exists
            os.makedirs(output_dir, exist_ok=True)

            # get filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{output_dir}/{func.__name__}_{timestamp}.prof"

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                profiler.dump_stats(filename)
                print(f"Profile saved to {filename}")
            return result

        return wrapper

    return decorator


# ---------- Instructions ---------

# 1. pip install snakeviz
# 2. add decorator @profileit() to your function
# 3. after running write in Terminal: snakeviz {path_to_your_profiler_.prof_file}

# --------- Example case --------
if __name__ == '__main__':
    @profileit()
    def my_function():
        for i in range(1000):
            sum([j for j in range(100)])


    my_function()
