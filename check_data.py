# check_data.py
import pandas as pd
import json

# Define the paths (assuming ./data is correct)
PARQUET_PATH = './data/task_a_trial.parquet'
LABEL_TO_ID_PATH = './data/label_to_id.json'

print("--- 1. Unique Labels in Parquet File ---")
try:
    # Read the parquet file
    df = pd.read_parquet(PARQUET_PATH)
    # Get all unique labels in the 'label' column and convert to string
    unique_data_labels = df['label'].astype(str).unique().tolist()
    print("Labels found in your data:", unique_data_labels)
except FileNotFoundError:
    print(f"Error: Parquet file not found at {PARQUET_PATH}")
    exit()

print("\n--- 2. Labels Defined in label_to_id.json ---")
try:
    # Load the defined labels
    with open(LABEL_TO_ID_PATH, 'r') as f:
        label_to_id = json.load(f)
    defined_labels = list(label_to_id.keys())
    print("Labels defined in JSON:", defined_labels)
except FileNotFoundError:
    print(f"Error: JSON file not found at {LABEL_TO_ID_PATH}")
    exit()

# Identify the discrepancy
discrepancies = [label for label in unique_data_labels if label not in defined_labels]

if discrepancies:
    print("\n--- 3. Discrepancy Found (MUST BE FIXED) ---")
    print("The following labels exist in your data but NOT in your JSON file:", discrepancies)
else:
    print("\nNo label discrepancy found. All good to proceed.")