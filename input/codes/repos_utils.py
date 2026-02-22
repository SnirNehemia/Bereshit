from pathlib import Path

import yaml


def fetch_directory(project_name: str = 'Bereshit'):
    project_folder = ""
    for parent in Path(__file__).parents:
        if parent.stem == project_name:
            project_folder = parent
            break

    return project_folder


def read_yaml(filepath):
    with open(filepath, 'r') as f:
        raw = yaml.safe_load(f)
    return raw


def write_yaml(filepath, data):
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f)


def get_data_from_config(config_name, folder_full_path: str | Path = ""):
    project_folder = fetch_directory()
    if folder_full_path == "":
        full_path = project_folder.joinpath("input").joinpath("yamls").joinpath(config_name)
    else:
        full_path = Path(folder_full_path).joinpath(config_name)
    data_dict = read_yaml(filepath=full_path)
    return data_dict, full_path
