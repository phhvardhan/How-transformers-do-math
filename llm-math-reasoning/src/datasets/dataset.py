import csv
import random

# Paths for dataset CSV files
basic_dataset_path = "basic_arithmetic_dataset.csv"
scratchpad_dataset_path = "scratchpad_dataset.csv"
advanced_dataset_path = "advanced_arithmetic_dataset.csv"

# Generate Basic Arithmetic Dataset (5,000 samples)
def generate_basic_arithmetic_dataset(file_path, num_samples):
    operations = ["+", "-", "*", "/"]

    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input", "Expected Output"])

        for _ in range(num_samples):
            num1 = random.randint(1, 99)
            num2 = random.randint(1, 99)
            operation = random.choice(operations)
            expression = f"{num1} {operation} {num2}"

            # Calculate the expected result
            if operation == "+":
                result = num1 + num2
            elif operation == "-":
                result = num1 - num2
            elif operation == "*":
                result = num1 * num2
            elif operation == "/":
                # Avoid division by zero and rounding issues
                if num2 != 0:
                    result = round(num1 / num2, 2)
                else:
                    continue

            csv_writer.writerow([expression, str(result)])

# Generate Scratchpad Dataset (8,000 samples)
def generate_scratchpad_dataset(file_path, num_samples):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input", "Expected Output"])

        for _ in range(num_samples):
            num1 = random.randint(100, 999)
            num2 = random.randint(100, 999)
            expression = f"Calculate {num1} + {num2} step-by-step"

            # Generate stepwise output
            step1 = f"Step 1: Break down {num1} into hundreds, tens, and units."
            step2 = f"Step 2: Break down {num2} into hundreds, tens, and units."
            step3 = f"Step 3: Add hundreds, add tens, add units."
            step4 = f"Step 4: Final result is {num1 + num2}."

            expected_output = f"{step1}\n{step2}\n{step3}\n{step4}"

            csv_writer.writerow([expression, expected_output])

# Generate Advanced Arithmetic Dataset (10,000 samples)
def generate_advanced_arithmetic_dataset(file_path, num_samples):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Input", "Expected Output"])

        for _ in range(num_samples):
            num1 = random.randint(1000, 9999)
            num2 = random.randint(1000, 9999)
            expression = f"Apply rules to calculate {num1} + {num2}"

            # Describe rule-based approach
            rule1 = "Rule 1: Start from the units place and move left."
            rule2 = "Rule 2: Carry over if sum exceeds 9."
            rule3 = f"Step-by-step calculation yields the result: {num1 + num2}"

            expected_output = f"{rule1}\n{rule2}\n{rule3}"

            csv_writer.writerow([expression, expected_output])

if __name__ == "__main__":
    # Generate datasets
    generate_basic_arithmetic_dataset(basic_dataset_path, 5000)
    generate_scratchpad_dataset(scratchpad_dataset_path, 8000)
    generate_advanced_arithmetic_dataset(advanced_dataset_path, 10000)

    print("Datasets have been generated successfully.")
