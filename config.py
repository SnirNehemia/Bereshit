import matplotlib.pyplot as plt
import matplotlib
import platform
from pathlib import Path

# make sure we can plot for debugging (did not test on debugger mode)
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

# for the video saving
project_folder = Path(__file__).parent
plt.rcParams['animation.ffmpeg_path'] = project_folder.joinpath(
    r'ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe')
