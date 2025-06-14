import cProfile
import functools
import os
from datetime import datetime
from pathlib import Path

from input.codes.repos_utils import fetch_directory

OUTPUT_DIR = fetch_directory().joinpath('profiles')
now = datetime.now()
DEFAULT_TIMESTAMP = now.strftime('%Y-%m-%d_T_%H-%M-%S')


def profileit(output_dir: Path = OUTPUT_DIR, timestamp: str = DEFAULT_TIMESTAMP):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # make sure directory exists
            os.makedirs(output_dir, exist_ok=True)

            # get filename with timestamp
            filename = f"{output_dir}/{timestamp}_{func.__name__}_profiler.prof"
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
