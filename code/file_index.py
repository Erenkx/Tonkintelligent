"""
Builds a project file index for agent use.
"""

import os
import json


IGNORE_SUFFIXES = ['.json', '.npy', '.index']


def build_file_index(
    project_root: str
) -> None:
    """
    Builds a file index for all projects in the given root directory.
    
    Args:
        project_root (str): The root directory containing project 
            folders.
    """
    file_index = {}

    for project_name in os.listdir(project_root):
        project_path = os.path.join(project_root, project_name)
        if not os.path.isdir(project_path):
            continue

        file_list = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.startswith('.'):
                    continue
                if any(file.endswith(suffix) for suffix in IGNORE_SUFFIXES):
                    continue
                
                relative_path = os.path.relpath(
                    os.path.join(root, file), project_root
                )
                file_list.append(relative_path)

        file_index[project_name] = file_list


    with open(os.path.join(project_root, 'file_index.json'), 'w') as fout:
        json.dump(file_index, fout, indent=4)
if __name__ == "__main__":
    project_root = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'projects'
    )
    build_file_index(project_root=project_root)
