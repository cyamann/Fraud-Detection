import pandas as pd
import random

def sample_csv(input_file, output_file, sample_size):
    try:
        # Counting total number of rows (excluding the header)
        print("Counting total number of rows...")
        with open(input_file, 'r') as file:
            n = sum(1 for line in file) - 1
        print(f"Total rows: {n}")

        # Ensure sample size is not greater than the total number of rows
        if sample_size > n:
            raise ValueError("Sample size cannot be greater than the total number of rows in the file")

        # Selecting rows to skip
        print("Selecting rows to skip...")
        skip = sorted(random.sample(range(1, n+1), n - sample_size))

        # Reading sampled data
        print("Reading sampled data...")
        df = pd.read_csv(input_file, skiprows=skip)

        # Writing sampled data to new file
        print("Writing sampled data to new file...")
        df.to_csv(output_file, index=False)

        print("Done. Sampled data written to the output file.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_csv = 'C:/Users/ozany/OneDrive/Belgeler/GitHub/Fraud-Detection/fraud-detection.csv'  # Path to the input CSV file
output_csv = 'C:/Users/ozany/OneDrive/Belgeler/GitHub/Fraud-Detection/new_csv.csv'  # Path to the output CSV file
sample_size = 10000  # Number of rows to sample

# Call the function
sample_csv(input_csv, output_csv, sample_size)
