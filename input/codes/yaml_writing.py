import yaml


def write_yaml(filepath, data):
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f)
