import yaml


def read_yaml(filepath):
    with open(filepath, 'r') as f:
        raw = yaml.safe_load(f)
    return raw
