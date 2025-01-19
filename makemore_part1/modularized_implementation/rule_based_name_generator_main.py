# This file implements a rule-based name generator model. 
# Please refer to 'makemore_part1/building_makemore_step_by_step/step_2_rule_based_name_generator.ipynb' for step
# by step explanation and execution of the implementation.

import argparse
import torch

from makemore_part1.modularized_implementation.constants import BOUNDARY_CHARACTER, SEED
from makemore_part1.modularized_implementation.utils import build_char_to_idx_map, build_idx_to_char_map, get_names_from_dataset
from torch import Tensor
from typing import List

def construct_probs(names: List[str]) -> Tensor:
    """Reads the names and constructs a probability distribution per character. Each probability 
    distribution gives the probability for each character to appear after the corresponding character.

    Args:
        names (List[str]): List of names.

    Returns:
        Tensor: Probability distributions for each character.
                SHAPE: (27, 27)
    """
    char_counts = torch.ones(size=(27, 27), dtype=torch.int32)
    char_to_idx_map = build_char_to_idx_map()
    for name in names:
        name = BOUNDARY_CHARACTER + name + BOUNDARY_CHARACTER
        for first_character, second_character in zip(name, name[1:]):
            first_character_idx = char_to_idx_map[first_character]
            second_character_idx = char_to_idx_map[second_character]
            char_counts[first_character_idx][second_character_idx] += 1
            char_probs = char_counts / char_counts.sum(dim=1, keepdim=True)
    return char_probs


def generate_name(char_probs: Tensor) -> str:
    """Generates a name using the probability distribution of characters.

    Args:
        char_probs (Tensor): Probability distributions for each character.
                             SHAPE: (27, 27)

    Returns:
        str: Generated name.
    """
    idx_to_char = build_idx_to_char_map()
    # Initially, the running name is empty.
    running_name = ""
    # We start with the bound_character (.) which is mapped to 0.
    predicted_char_idx = 0
    while True:    
        # Extract the probability distribution corresponding to the current character.
        current_char_probs = char_probs[predicted_char_idx]
        # Sample the next character based on the retrieved probability distribution in the previous step.
        predicted_char_idx = torch.multinomial(current_char_probs, num_samples=1).item()
        # If the predicted character is the bound_character, we break out of the loop.
        if predicted_char_idx == 0:
            break
        # Append the predicted character to the running name.
        running_name += idx_to_char[predicted_char_idx]
    # Print the generated name.
    return running_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based name generator")
    parser.add_argument("--num_names_to_generate", type=str, default="20", help="Number of names to generate")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for reproducibility")
    parser.parse_args()

    # Set the seed for reproducibility.
    torch.manual_seed(parser.parse_args().seed)

    # Get the names from the dataset.
    names = get_names_from_dataset()
    # Construct the probability distribution for each character.
    char_probs = construct_probs(names=names)
    # Generate the names.
    for _ in range(int(parser.parse_args().num_names_to_generate)):
        print(generate_name(char_probs))
