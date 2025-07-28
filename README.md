# LLM Mathematical Reasoning: Adaptive Hybrid Model

Adaptive arithmetic reasoning in large language models using a hybrid of case-based and rule-based logic, with reinforcement learning for iterative error correction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Team](#team)
- [References](#references)
- [License](#license)

## Overview

This project investigates mathematical reasoning in large language models by introducing an adaptive hybrid model. The model dynamically switches between case-based and rule-based logic for arithmetic tasks and employs reinforcement learning (PPO) for iterative error correction, achieving robust performance and high accuracy on complex arithmetic.

## Features

- Adaptive selection between case-based and rule-based reasoning
- Reinforcement learning error correction loop
- Generalizes well to unseen arithmetic problems
- Achieves high accuracy on multi-digit addition tasks

## Tech Stack

- Python 3.8+
- Hugging Face Transformers
- PyTorch
- PPO (Proximal Policy Optimization)
- Jupyter Notebooks

## Project Structure
llm-math-reasoning/         # <-- This is your repo's root folder
  ├── src/                  # Main source code directory (Python scripts go here)
  │   ├── datasets/         # Any scripts or CSVs for data loading/preprocessing
  │   ├── models/           # Python files for model definitions
  │   └── utils/            # Utility/helper scripts (functions, configs)
  ├── notebooks/            # Jupyter notebooks (.ipynb) for experiments/demos
  ├── reports/              # Project report, presentation slides, related PDFs
  ├── pretrained_models/    # Saved model checkpoints, tokenizer files, etc.
  ├── assets/               # Images/GIFs for your README or documentation
  ├── requirements.txt      # File listing all dependencies (for pip)
  ├── README.md             # This README file
  └── LICENSE               # Project license file

## Quick Start

Clone this repo:
``bash 
git clone https://github.com/phhvardhan/How-transformers-do-math.git
cd llm-math-reasoning

# Install dependencies:
pip install -r requirements.txt

# Run scripts or open notebooks in /notebooks to try experiments

## Results
  Achieved 93–98% accuracy on multi-digit addition (Final Report)

  Outperformed baseline transformer models on both efficiency and accuracy

  Robust error correction for complex arithmetic problems

## Team
Sasi Kiran Boyapati
Veera Venkata Satya Sai Bhargavi Manda
Hema Harsha Vardhan Peela
Nithish Chandra Vyas Talluru
Priscilla Grace Kolagani

