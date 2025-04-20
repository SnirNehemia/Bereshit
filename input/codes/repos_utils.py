from pathlib import Path


def fetch_directory(project_name: str = 'Bereshit'):
    project_folder = ""
    for parent in Path(__file__).parents:
        if parent.stem == project_name:
            project_folder = parent
            break

    return project_folder
