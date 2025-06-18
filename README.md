# Bereshit
## Installation
- install ffmpeg and put it in project folder
- create venv:
  - py -3.10 -m venv /.venv
  - .venv\Scripts\activate
  - pip3 install -r requirements.txt

## Config and physical model
- stored in "input/yamls" folder
- Config contain all relevant parameters for simulation
- Physical model contain all physical constants and most rules (some may be in Creature class)
- Environment is loaded from a PNG file stated in config file (example: "Penvs\Env1.png") with the following format:
- black areas are forbidden areas
- yellow areas are low vegetation areas (grass)
- green areas are high vegetation areas (trees)


## How to use profiler
- pip install snakeviz 
- add decorator @profileit() to your function
  - add output_dir and timestamp if you wish (else it will be saved in 'profiles' folder with timestamp after completion).
- to visualize results write in Terminal: snakeviz {path_to_your_prof_file}

## How to load json outputs
- see "load_statistics_logs_from_json.py" for an example.

