import argparse
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load the dataset
def load_dataset(file_path):
    # Load CSV dataset using pandas
    return pd.read_csv(file_path)

# Fine-tuning function
def fine_tune_model(model, tokenizer, train_dataset, mode):
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

# Main function to fine-tune the model in different modes
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

    # Train the model using the specified mode
    fine_tune_model(model, tokenizer, dataset, args.mode)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt3.5", "llama"], required=True, help="Choose the model: gpt3.5 or llama")
    parser.add_argument("--mode", choices=["direct", "scratchpad", "rfft"], required=True, help="Choose the fine-tuning mode: direct, scratchpad, or rfft")
    args = parser.parse_args()

    # Run the main function
    main(args)
