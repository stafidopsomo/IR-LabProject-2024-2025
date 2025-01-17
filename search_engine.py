import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

# Load documents from CSV
def load_documents(file_path="processed_articles.csv"):
    try:
        df = pd.read_csv(file_path)
        df['stemmed_tokens'] = df['stemmed_tokens'].apply(eval)
        titles = df['title'].tolist()
        documents = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens)).tolist()
        return titles, documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        exit()

# Load inverted index
def load_inverted_index(file_path="inverted_index.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: inverted_index.json not found.")
        exit()

# Boolean Retrieval (Now applies `NOT` before returning results)
def boolean_search(query, inverted_index):
    terms = query.split()
    result_docs = set()

    def get_docs(term):
        return set(inverted_index.get(term, []))

    current_docs = set()
    operation = None

    for term in terms:
        if term.upper() in ["AND", "OR", "NOT"]:
            operation = term.upper()
        else:
            docs = get_docs(term)
            if operation == "AND":
                current_docs &= docs
            elif operation == "OR":
                current_docs |= docs
            elif operation == "NOT":
                current_docs -= docs
            else:
                current_docs = docs

    return current_docs

# TF-IDF Retrieval (Filters excluded docs before ranking)
def tfidf_retrieval(query, documents, inverted_index, titles):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    scores = (doc_vectors @ query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(scores)[::-1]
    
    # Get Boolean filter
    allowed_docs = boolean_search(query, inverted_index)

    # Filter out disallowed documents
    ranked_indices = [i for i in ranked_indices if titles[i] in allowed_docs]

    return ranked_indices, scores[ranked_indices]

# BM25 Retrieval (Filters excluded docs before ranking)
def bm25_retrieval(query, documents, inverted_index, titles):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = query.split()
    
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1]

    # Get Boolean filter
    allowed_docs = boolean_search(query, inverted_index)

    # Filter out disallowed documents
    ranked_indices = [i for i in ranked_indices if titles[i] in allowed_docs]

    return ranked_indices, scores[ranked_indices]
