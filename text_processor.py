"""
Dimitrakopoulos Stylianos 
AM: 18390149
Προγραμμα Σπουδων ΠΑΔΑ
"""

import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Φoρτωση των άρθρων από  JSON
with open('wikipedia_articles.json', 'r', encoding='utf-8') as f:
    articles_data = json.load(f)

# Μετατροπή των δεδομένων σε DataFrame
articles_df = pd.DataFrame(articles_data)

# Συνάρτηση για την εξαγωγή λέξεων από το κείμενο
def extract_tokens(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

# tokenization
articles_df['tokens'] = articles_df['content'].apply(extract_tokens)

nltk.download('stopwords')
nltk.download('punkt')

# Αφαίρεση stop words
stop_words_set = set(stopwords.words('english'))
articles_df['filtered_tokens'] = articles_df['tokens'].apply(
    lambda words: [word for word in words if word not in stop_words_set]
)

# stemming
stemmer = PorterStemmer()
articles_df['stemmed_tokens'] = articles_df['filtered_tokens'].apply(
    lambda words: [stemmer.stem(word) for word in words]
)

# Αποθήκευση του επεξεργασμένου συνόλου δεδομένων σε CSV
articles_df[['title', 'stemmed_tokens']].to_csv('processed_articles.csv', index=False)
print("Processing complete. Data saved to 'processed_articles.csv'.")
