import matplotlib.pyplot as plt
import matplotlib
import platform

# make sure we can plot for debugging (did not test on debugger mode)
import matplotlib
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')

# for the video saving
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\snirn\Documents\GitHub\Bereshit\ffmpeg-2025-02-20-git-bc1a3bfd2c-essentials_build\bin\ffmpeg.exe'