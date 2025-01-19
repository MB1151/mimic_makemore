# This file implements a neural_network-based name generator model. 
# Please refer to 'makemore_part1/building_makemore_step_by_step/step_5_rule_based_name_generator.ipynb' for step
# by step explanation and execution of the implementation.

import argparse
import torch
import torch.nn.functional as F

from makemore_part1.modularized_implementation.constants import BOUNDARY_CHARACTER, LEARNING_RATE, NUM_CHARACTERS, NUM_EPOCHS, SEED
from makemore_part1.modularized_implementation.utils import build_char_to_idx_map, build_idx_to_char_map, get_names_from_dataset
from torch import Tensor
from typing import List, Tuple


def convert_names_to_model_data(names: List[str]) -> Tuple[Tensor, Tensor]:
    """Constructs input-output character pairs from the names. The input is the first character and the 
    output is the following character. The input is further encoded using one-hot encoding. The inputs
    and outputs are returned as tensors.

    Args:
        names (List[str]): List of names

    Returns:
        Tuple[Tensor, Tensor]: Encoded inputs and targets.
                               SHAPE of encoded inputs: (num_pairs, 27)
                               SHAPE of targets: (num_pairs,)
    """
    input_list = []
    target_list = []
    char_to_idx = build_char_to_idx_map()
    for name in names:
        name = BOUNDARY_CHARACTER + name + BOUNDARY_CHARACTER
        for first_char, second_char in zip(name, name[1:]):
            first_char_idx = char_to_idx[first_char]
            second_char_idx = char_to_idx[second_char]
            input_list.append(first_char_idx)
            target_list.append(second_char_idx)
    inputs = torch.tensor(data=input_list, dtype=torch.int64)
    encoded_inputs = F.one_hot(inputs, num_classes=NUM_CHARACTERS).float()
    targets = torch.tensor(data=target_list, dtype=torch.int64)
    return encoded_inputs, targets


def train_model(encoded_inputs: Tensor, targets: Tensor, num_epochs: int=NUM_EPOCHS) -> Tensor:
    """Trains the neural network model to predict the next character given the current character.
    The model is trained using the negative log-likelihood loss. We currently don't use any Pytorch
    modules to build the model. We directly use the model weights and perform forward and backward
    propagation.

    Args:
        encoded_inputs (Tensor): encoded inputs using one-hot encoding (27 characters or classes).
                                 SHAPE: (num_pairs, 27)
        targets (Tensor): Tensor containing the target character indices.
                          SHAPE: (num_pairs,)
        num_epochs (int, optional): _description_. Defaults to NUM_EPOCHS.

    Returns:
        Tensor: Returns the trained model weights.
    """
    model_weights = torch.randn(size=(27, 27), dtype=torch.float32, requires_grad=True)
    for iteration in range(num_epochs):
        # Forward Propagation
        model_output = encoded_inputs @ model_weights
        logits = model_output.exp()
        probs = logits / logits.sum(dim=1, keepdim=True)
        target_probs = probs[torch.arange(start=0, end=len(targets)), targets]
        loss = -torch.log(target_probs).mean()
        print(f"Loss after iteration {iteration} is {loss.item()}")
        # Back Propagation
        # Always, zero the weights from the previous loop. Other it will update the gradients instead of over-writing.
        model_weights.grad = None
        loss.backward()
        model_weights.data += -LEARNING_RATE * model_weights.grad
    return model_weights


def generate_name(model_weights: Tensor) -> str:
    """Generates a name using the trained model weights.

    Args:
        model_weights (Tensor): Trained model weights.
                                SHAPE: (27, 27)

    Returns:
        str: Generated name.
    """
    name = BOUNDARY_CHARACTER
    char_to_idx = build_char_to_idx_map()
    idx_to_char = build_idx_to_char_map()
    while True:
        prev_char_idx = char_to_idx[name[-1]]
        input_tensor = torch.zeros(size=(1, 27), dtype=torch.float32)
        input_tensor[0][prev_char_idx] = 1.0
        output_tensor = input_tensor @ model_weights
        logits = output_tensor.exp()
        output_probs = logits / logits.sum(dim=1, keepdim=True)
        output_char = idx_to_char[torch.multinomial(output_probs, num_samples=1).item()]
        if output_char == BOUNDARY_CHARACTER:
            break
        name += output_char
    return name[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_names_to_generate", type=int, default=20, help="Number of names to generate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for reproducibility.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs to train the model.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    names = get_names_from_dataset()
    encoded_inputs, targets = convert_names_to_model_data(names=names)
    model_weights = train_model(encoded_inputs=encoded_inputs, targets=targets, num_epochs=args.num_epochs)
    for _ in range(args.num_names_to_generate):
        print(generate_name(model_weights))