# Mimic Andrej Karpathy's makemore (IN PROGRESS)

This repository implements the models built in Andrej Karpathy's makemore 
series specifically [`building makemore`](https://youtu.be/PaCmpygFfXo?si=JlySFwJ-DxEbvOQt) and [`building makemore part 2`](https://youtu.be/TCH_1BHY58I?si=y7qEaty70xi2fsnX). These models generate new Indian names by learning patterns from an [Indian name dataset](makemore_part1/Data/IndianNamesUnique.csv).

Please navigate to his original videos to understand more about these models and circle back to this repository.


## Table Of Contents

- [Useful Resources](#useful-resources)
- [Repository Structure](#repository-structure)
- [Usage](#usage)


## Useful Resources

- <u>[Video](https://youtu.be/PaCmpygFfXo?si=JlySFwJ-DxEbvOQt)</u> giving a walk through of building simple name generation models implemented in [makemore_part1](makemore_part1) &rarr; *By Andrej Karpathy*
- <u>[Video](https://youtu.be/TCH_1BHY58I?si=y7qEaty70xi2fsnX)</u> giving a walk through of building a name generation model implemented in [makemore_part2](makemore_part1) &rarr; *By Andrej Karpathy*
- <u>[Video](https://www.youtube.com/watch?v=ScduwntrMzc&t=41s)</u> explains the concepts of Likelihood and several related concepts &rarr; *By zedstatistics*
- <u>[Video](https://www.youtube.com/watch?v=7kLHJ-F33GI&t=40s)</u> explains the concepts of Maximum Likelihood Estimation &rarr; *By zedstatistics*
- <u>[Paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)</u> introducing a neural network architecture for language modeling &rarr; *By Yoshua Bengio*
- <u>[Paper](https://arxiv.org/pdf/1506.01186)</u> that first introduced Learning Rate Range Test &rarr; *By Leslie N. Smith*
- <u>[Blog](https://brandonmorris.dev/2018/06/24/mastering-the-learning-rate/)</u> explains how to choose good learning rate for training neural networks &rarr; *By Brandon Morris*
- <u>[Blog](https://blog.dataiku.com/the-learning-rate-finder-technique-how-reliable-is-it)</u> provides a few recommendations in choosing learning rate bounds and the reliability of LRRT methodology.


## Repository Structure

This repository contains the implementation of multiple models from the Andrej Karpathy's makemore series.

[`makemore_part1`](makemore_part1/) implements two approaches to generate new names based on an Indian name dataset.
- First Model &rarr; A statistical name generator based on probablistic rules without any language modeling.
- Second Model &rarr; A neural network model equivalent to the above rule based probablistic model.

[`makemore_part2`](makemore_part2/) implements a basic language model to generate names based on the neural network architecture introduced by `Yoshua Bengio` in this [paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).


Each `makemore` directory contains two parallel sets of implementations for the same models:

1. Step by Step implementation in the Jupiter notebooks where each line of code is explained in detail and executed.
2. Modularized implementation of the same model in python scripts which is used to train a a very small toy dataset.

---

### `makemore_part1/`

#### [`makemore_part1/building_makemore_step_by_step/`](makemore_part1/building_makemore_step_by_step/)

This directory contains step by step implementation, explanations, and execution for name generator models built in `makemore_part1`

- [`step_1_data_exploration.ipynb`](makemore_part1/building_makemore_step_by_step/step_1_data_exploration.ipynb)
    * Explores the names dataset and sanitizes it to be used with our name generator models.
- [`step_2_rule_based_name_generator.ipynb`](makemore_part1/building_makemore_step_by_step/step_2_rule_based_name_generator.ipynb)
    * Shows how to build a rule based probablistic model to generate new India names given a names dataset.
- [`step_3_model_quality.ipynb`](makemore_part1/building_makemore_step_by_step/step_3_model_quality.ipynb)
    * Shows to evaluate the quality of a name generator model and model a loss function using likelihood function.
- [`step_4_rule_based_model_quality.ipynb`](makemore_part1/building_makemore_step_by_step/step_4_rule_based_model_quality.ipynb)
    * Evaluates the quality of the rule based probabilistic model using log likelihood loss.
- [`step_5_neural_network_based_name_generator.ipynb`](makemore_part1/building_makemore_step_by_step/step_5_neural_network_based_name_generator.ipynb)
    * Shows how to build a neural network model equivalent to the rule based probablistic name generator model.

#### [`makemore_part1/Data`](makemore_part1/Data/)

This directory contains the dataset used by both models in `makemore_part1` and `makemore_part2` implementations.

- [`IndianNamesUnique.csv`](makemore_part1/Data/IndianNamesUnique.csv)
    * Dataset containing Indian names downloaded from [Kaggle](https://www.kaggle.com/datasets/surajpratap/sixty-thousand-unique-indian-names-dataset?resource=download).
- [`names.text`](makemore_part1/Data/names.txt)
    * Sanitized names dataset used to train the models in `makemore_part1` and `makemore_part2`


#### [`makemore_part1/modularized_implementation`](makemore_part1/modularized_implementation/)

This directory houses the python scripts containing the same models built in the Jupyter notebooks but modularized to be reused for training and quick experimentation.

--- 

### `makemore_part2/`

#### [`makemore_part2/building_makemore_step_by_step/`](makemore_part2/building_makemore_step_by_step/)

This directory contains step by step implementation, explanations, and execution for name generator models built in `makemore_part2`

- [`step_1_data_creation.ipynb`](makemore_part2/building_makemore_step_by_step/step_1_data_creation.ipynb)
    * Explains the neural network architecture proposed by `Yoshua Bengio` in this [paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
    * Explains how the above architecture is converted to be used for name generation and input-output formats required by our model.
    * Shows how to create the data in the format required by our name generator model.
- [`step_2_model_creation.ipynb`](makemore_part2/building_makemore_step_by_step/step_2_model_creation.ipynb)
    * Shows how to build the model, the loss function and does a sample training loop.
- [`step_3_model_training.ipynb`](makemore_part2/building_makemore_step_by_step/step_3_model_training.ipynb)
    * Shows how to train the language model to generate names and compares the quality with the simple models built in [`makemore_part1`](makemore_part1/).
    * Also, explains how to choose optimal learning rates for a deep learning model and specifically for this language model.


#### [`makemore_part2/Data`](makemore_part2/Data/)

This directory contains the images used in the `makemore_part2` jupiter notebooks to explain the neural network architecture and related concepts.


#### [`makemore_part2/modularized_implementation`](makemore_part2/modularized_implementation/)

YET TO BE IMPLEMENTED


## Usage

### Setup

Create a Virtual Environment for this project that will contain all the dependencies.

```python3 -m venv .makemore_venv```

Run the following command to install the necessary packages in the virtual environment.

```pip install -r requirements.txt```

---

### Running makemore_part1

1) You can go through the Jupiter notebooks to execute and understand each line of code.
2) You can run the python scripts to build the name generator models.

Run the following command to build the rule-based probablistic name generator model:

```python3 makemore_part1/modularized_implementation/rule_based_name_generator_main.py  ```

Run the following command to build the equivalent neural network based name generator model:

```python3 makemore_part1/modularized_implementation/neural_network_based_name_generator_main.py```


--- 

### Running makemore_part2

1) You can go through the Jupiter notebooks to execute and understand each line of code.
2) You can run the python scripts to build the name generator models.

YET TO BE ADDED