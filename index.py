"""
Dimitrakopoulos Stylianos 
AM: 18390149
Προγραμμα Σπουδων ΠΑΔΑ
"""

import pandas as pd
from collections import defaultdict
import json
import re

# Eξαγωγη λέξεων από το κείμενο
def process_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# Δημιουργία inverted index
def generate_index(data):
    index = defaultdict(set)
    
    for _, row in data.iterrows():
        doc_title = row['title']
        words = process_text(row['content'])

        for word in words:
            index[word].add(doc_title)

    return {term: list(docs) for term, docs in index.items()}

# Φόρτωση του επεξεργασμένου αρχείου
try:
    articles_data = pd.read_csv("processed_articles.csv")
    articles_data['content'] = articles_data['stemmed_tokens'].apply(eval).apply(' '.join)
    print("Processed data loaded successfully.")
except FileNotFoundError:
    print("Error: 'processed_articles.csv' not found.")
    exit()

# Κατασκευή και αποθήκευση του inverted index
print("Building the inverted index...")
inverted_index = generate_index(articles_data)
print("Inverted index created.")

index_file = "inverted_index.json"
with open(index_file, "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=4)

print(f"Inverted index saved to '{index_file}'.")
