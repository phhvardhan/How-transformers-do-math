import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load test dataset
def load_test_dataset(file_path):
    # Load CSV dataset using pandas
    return pd.read_csv(file_path)

# Evaluation function for adaptive hybrid model
def evaluate_model(model, tokenizer, dataset):
    correct = 0
    total = len(dataset)
    
    for _, row in dataset.iterrows():
        input_text = row['Input']
        expected_output = row['Expected Output']

        # Tokenize input and generate output
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if the output matches expected answer
        if predicted_output.strip() == expected_output.strip():
            correct += 1
    
    return correct / total

# Main function for evaluation
def main(args):
    # Load the model and tokenizer
    model_name = "gpt3.5-model-name" if args.model == "gpt3.5" else "llama-7b-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(f"./results_{args.mode}/checkpoint-best")

    # Load test dataset
    test_dataset_path = "test_arithmetic_dataset.csv"
    test_dataset = load_test_dataset(test_dataset_path)

    # Evaluate the model
    accuracy = evaluate_model(model, tokenizer, test_dataset)
    print(f"Accuracy for {args.model} ({args.mode} mode with hybrid and error correction): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt3.5", "llama"], required=True, help="Choose the model: gpt3.5 or llama")
    parser.add_argument("--mode", choices=["direct", "scratchpad", "rfft"], required=True, help="Choose the evaluation mode: direct, scratchpad, or rfft")
    args = parser.parse_args()

    # Run the main function
    main(args)
