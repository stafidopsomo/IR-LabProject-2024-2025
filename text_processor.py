import json
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the JSON file
with open('wikipedia_articles.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Custom tokenizer function
def custom_tokenizer(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Apply tokenization
df['tokens'] = df['content'].apply(custom_tokenizer)

# Download NLTK data (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Remove stop words
stop_words = set(stopwords.words('english'))
df['filtered_tokens'] = df['tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

# Apply stemming
ps = PorterStemmer()
df['stemmed_tokens'] = df['filtered_tokens'].apply(
    lambda tokens: [ps.stem(word) for word in tokens]
)

# Save the processed data
df[['title', 'stemmed_tokens']].to_csv('processed_articles.csv', index=False)
print("Processing complete. Data saved to 'processed_articles.csv'.")
