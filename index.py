import pandas as pd
from collections import defaultdict
import json
import re

# Custom tokenizer function
def custom_tokenizer(text):
    tokens = re.findall(r'\b\w+\b', text.lower())  # Include words and numbers
    return tokens

# Function to build the inverted index
def build_inverted_index(df):
    inverted_index = defaultdict(set)  # Use a set to avoid duplicates

    # Iterate through the DataFrame
    for _, row in df.iterrows():
        title = row['title']
        tokens = custom_tokenizer(row['content'])  # Apply updated tokenizer
        
        # Add each token to the index
        for token in tokens:
            inverted_index[token].add(title)

    # Convert sets to lists for final output
    return {key: list(value) for key, value in inverted_index.items()}

# Load the processed CSV file
try:
    processed_df = pd.read_csv("processed_articles.csv")
    processed_df['content'] = processed_df['stemmed_tokens'].apply(eval).apply(' '.join)  # Join tokens into content
    print("Processed data loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'processed_articles.csv' was not found.")
    exit()

# Build the index
print("Building the inverted index...")
inverted_index = build_inverted_index(processed_df)
print("Inverted index built successfully.")

# Save the inverted index to a JSON file
output_file = "inverted_index.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=4)
print(f"Inverted index saved to '{output_file}'.")
