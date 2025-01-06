import pandas as pd
from collections import defaultdict
import json

# Step 1: Load the processed CSV file
try:
    processed_df = pd.read_csv("processed_articles.csv")
    print("Processed data loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'processed_articles.csv' was not found.")
    exit()

# Step 2: Convert 'stemmed_tokens' from string to list
processed_df['stemmed_tokens'] = processed_df['stemmed_tokens'].apply(lambda x: eval(x))
print("Converted 'stemmed_tokens' column to list format.")

# Step 3: Function to build the inverted index
def build_inverted_index(df):
    inverted_index = defaultdict(set)  # Use a set to avoid duplicates

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        title = row['title']  # Document title
        tokens = row['stemmed_tokens']  # Access tokens as a list

        # Add each token to the index
        for token in tokens:
            inverted_index[token].add(title)

    # Convert sets to lists for final output
    return {key: list(value) for key, value in inverted_index.items()}

# Step 4: Build the inverted index
inverted_index = build_inverted_index(processed_df)
print("Inverted index built successfully.")

# Step 5: Save the inverted index to a JSON file
output_file = "inverted_index.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=4)
print(f"Inverted index saved to '{output_file}'.")

# Optional: Display a sample of the inverted index
sample_keys = list(inverted_index.keys())[:10]  # Get first 10 keys
sample_index = {key: inverted_index[key] for key in sample_keys}
print("Sample of the inverted index:")
print(json.dumps(sample_index, indent=4))
