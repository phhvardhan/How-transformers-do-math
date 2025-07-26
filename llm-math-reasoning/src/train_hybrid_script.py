
import argparse
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, PPOTrainer, TrainingArguments
import torch

# Load the dataset
def load_dataset(file_path):
    # Load CSV dataset using pandas
    return pd.read_csv(file_path)

# Calculate median digit length for a batch
def calculate_median_length(dataset):
    digit_lengths = dataset['Input'].apply(lambda x: len(x.split()))
    sorted_lengths = digit_lengths.sort_values().to_list()
    n = len(sorted_lengths)

    if n % 2 == 1:
        median = sorted_lengths[n // 2]
    else:
        median = (sorted_lengths[(n // 2) - 1] + sorted_lengths[n // 2]) / 2

    return median

# Adaptive Hybrid Approach - choose approach based on median threshold
def choose_approach(input_text, median_length):
    if len(input_text.split()) <= median_length:
        return "case_based"
    else:
        return "rule_based"

# Fine-tuning function with adaptive hybrid and error correction
def fine_tune_model(model, tokenizer, train_dataset, mode):
    # Calculate the median length for dynamic threshold
    median_length = calculate_median_length(train_dataset)

    # Prepare the dataset for training
    def tokenize_function(examples):
        return tokenizer(examples["Input"], truncation=True, padding=True, max_length=128)

    # Tokenize dataset
    tokenized_dataset = train_dataset.apply(lambda row: tokenize_function({'Input': row['Input']}), axis=1)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{mode}",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=2,
        save_steps=500,
        logging_dir=f"./logs_{mode}",
    )

    # Trainer Setup
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train the model with hybrid approach and error correction
    for index, row in train_dataset.iterrows():
        input_text = row['Input']
        expected_output = row['Expected Output']
        approach = choose_approach(input_text, median_length)

        # Perform training based on approach
        if approach == "case_based":
            trainer.train_step(input_text, expected_output, mode="case_based")
        elif approach == "rule_based":
            trainer.train_step(input_text, expected_output, mode="rule_based")

    # Save the model
    trainer.save_model()

# Main function to fine-tune the model in different modes with improvements
def main(args):
    # Load the pre-trained model and tokenizer
    model_name = "gpt3.5-model-name" if args.model == "gpt3.5" else "llama-7b-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load dataset
    dataset_path = {
        "direct": "basic_arithmetic_dataset.csv",
        "scratchpad": "scratchpad_dataset.csv",
        "rfft": "advanced_arithmetic_dataset.csv"
    }[args.mode]

    dataset = load_dataset(dataset_path)

    # Train the model using the specified mode with hybrid and error correction
    fine_tune_model(model, tokenizer, dataset, args.mode)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt3.5", "llama"], required=True, help="Choose the model: gpt3.5 or llama")
    parser.add_argument("--mode", choices=["direct", "scratchpad", "rfft"], required=True, help="Choose the fine-tuning mode: direct, scratchpad, or rfft")
    args = parser.parse_args()

    # Run the main function
    main(args)
