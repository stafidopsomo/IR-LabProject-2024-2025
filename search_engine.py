import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

# Load the documents from processed_articles.csv
def load_documents(file_path="processed_articles.csv"):
    try:
        df = pd.read_csv(file_path)

        # Ensure 'stemmed_tokens' column is parsed correctly
        df['stemmed_tokens'] = df['stemmed_tokens'].apply(eval)

        # Convert tokens to a single string for each document
        titles = df['title'].tolist()
        documents = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens)).tolist()
        return titles, documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        exit()

# Load the inverted index
def load_inverted_index(file_path="inverted_index.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: inverted_index.json not found. Please ensure it exists.")
        exit()

# Boolean retrieval
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
            operation = None

    result_docs = current_docs
    return result_docs

# Helper function to apply Boolean logic to filter documents
def filter_documents(query, inverted_index, titles):
    query_terms = query.split()

    # Identify required and excluded terms
    required_terms = [term for term in query_terms if term.upper() not in ["AND", "OR", "NOT"] and not term.upper().startswith("NOT")]
    excluded_terms = [term[4:] for term in query_terms if term.upper().startswith("NOT")]

    # Filter documents by required terms (AND logic)
    candidate_indices = set(range(len(titles)))
    for term in required_terms:
        if term in inverted_index:
            term_indices = {titles.index(doc) for doc in inverted_index[term]}
            candidate_indices &= term_indices  # Intersect for AND logic
        else:
            candidate_indices = set()  # If a term is missing, no documents match
            break

    # Exclude documents containing unwanted terms
    excluded_indices = set()
    for term in excluded_terms:
        if term in inverted_index:
            excluded_indices |= {titles.index(doc) for doc in inverted_index[term]}
    candidate_indices -= excluded_indices

    return candidate_indices

# TF-IDF retrieval with Boolean filtering
def tfidf_retrieval(query, documents, inverted_index, titles):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Filter documents using Boolean logic
    filtered_indices = list(filter_documents(query, inverted_index, titles))  # Convert to list

    if not filtered_indices:
        return [], []

    # Restrict TF-IDF computation to filtered indices
    filtered_docs = [documents[idx] for idx in filtered_indices]
    filtered_tfidf_matrix = vectorizer.fit_transform(filtered_docs)

    # Rank remaining candidates using TF-IDF scores
    query_terms = [term for term in query.split() if term.upper() not in ["AND", "OR", "NOT"]]
    query_vec = vectorizer.transform([" ".join(query_terms)])
    scores = np.dot(filtered_tfidf_matrix, query_vec.T).toarray().flatten()

    ranked_indices = np.argsort(scores)[::-1]  # Sort by scores in descending order

    # Map filtered indices back to original document indices
    final_ranked_indices = [filtered_indices[idx] for idx in ranked_indices]
    return final_ranked_indices, [scores[idx] for idx in ranked_indices]

# BM25 retrieval with Boolean filtering
def bm25_retrieval(query, documents, inverted_index, titles):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Filter documents using Boolean logic
    filtered_indices = list(filter_documents(query, inverted_index, titles))  # Convert to list

    if not filtered_indices:
        return [], []

    # Restrict BM25 computation to filtered indices
    filtered_docs = [documents[idx] for idx in filtered_indices]
    filtered_bm25 = BM25Okapi([doc.split() for doc in filtered_docs])

    # Compute BM25 scores for the filtered subset
    query_terms = [term for term in query.split() if term.upper() not in ["AND", "OR", "NOT"]]
    scores = filtered_bm25.get_scores(" ".join(query_terms).split())

    ranked_indices = np.argsort(scores)[::-1]  # Sort by scores in descending order

    # Map filtered indices back to original document indices
    final_ranked_indices = [filtered_indices[idx] for idx in ranked_indices]
    return final_ranked_indices, [scores[idx] for idx in ranked_indices]

# Main function for user interaction
def main():
    print("Loading documents and inverted index...")
    titles, documents = load_documents("processed_articles.csv")
    inverted_index = load_inverted_index("inverted_index.json")
    print("Documents and inverted index loaded successfully!")

    print("\nSelect a retrieval model:")
    print("1. Boolean Retrieval")
    print("2. TF-IDF")
    print("3. Okapi BM25")

    while True:
        choice = input("\nEnter your choice (1/2/3 or 'exit' to quit): ").strip()
        if choice.lower() == "exit":
            break

        query = input("Enter your query: ").strip()
        if choice == "1":
            results = boolean_search(query, inverted_index)
            print(f"\nBoolean Retrieval Results:\n{results}")
        elif choice == "2":
            ranked_indices, scores = tfidf_retrieval(query, documents, inverted_index, titles)
            print("\nTF-IDF Retrieval Results:")
            for idx in ranked_indices[:10]:  # Show top 10 results
                print(f"Document: {titles[idx]}, Score: {scores[idx]:.4f}")
        elif choice == "3":
            ranked_indices, scores = bm25_retrieval(query, documents, inverted_index, titles)
            print("\nOkapi BM25 Retrieval Results:")
            for idx in ranked_indices[:10]:  # Show top 10 results
                print(f"Document: {titles[idx]}, Score: {scores[idx]:.4f}")
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
