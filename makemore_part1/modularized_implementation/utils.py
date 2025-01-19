import os
import string

from makemore_part1.modularized_implementation.constants import BOUNDARY_CHARACTER, NAMES_DATASET_PATH
from typing import Dict, List

# Returns the absolute path if relative path wrt to the repository is provided.
def get_absolute_path(relative_path: str) -> str:
    """Returns the absolute path if the relative path with respect to the repository is provided.

    Args:
        relative_path (str): Relative path with respect to the repository.

    Returns:
        str: Returns the absolute path on the machine.
    """
    # Please note that the implementation of this function depends on the placement of this file and
    # it might not work if this file is moved to a different location.
    cur_path = os.path.dirname(os.path.abspath(__file__))
    project_directory, _ = os.path.split(cur_path)
    repo_root, _ = os.path.split(project_directory)
    return os.path.join(repo_root, relative_path)


def get_names_from_dataset() -> List[str]:
    """Reads the names from the dataset file and returns them as a list.

    Returns:
        _type_: _description_
    """
    absolute_dataset_path = get_absolute_path(relative_path=NAMES_DATASET_PATH)   
    with open(absolute_dataset_path, "r") as file:
        names = [name.strip() for name in file.readlines()]
    return names


def build_char_to_idx_map() -> Dict[str, int]:
    """Builds a dictionary that maps characters to their corresponding index.

    Returns:
        dict: Dictionary mapping characters to their corresponding index.
    """
    char_to_idx = {char: idx + 1 for idx, char in enumerate(string.ascii_lowercase)}
    char_to_idx[BOUNDARY_CHARACTER] = 0
    return char_to_idx


def build_idx_to_char_map() -> Dict[int, str]:
    """Builds a dictionary that maps indices to their corresponding character.

    Returns:
        dict: Dictionary mapping indices to their corresponding character.
    """
    idx_to_char = {idx + 1: char for idx, char in enumerate(string.ascii_lowercase)}
    idx_to_char[0] = BOUNDARY_CHARACTER
    return idx_to_char