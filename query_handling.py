"""
Dimitrakopoulos Stylianos 
AM: 18390149
Προγραμμα Σπουδων ΠΑΔΑ
"""

import json

# Φόρτωση του inverted indexx
def load_index(file_path="inverted_index.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Εκτέλεση Boolean search
def boolean_retrieval(query, index_data):
    terms = query.split()
    results = set()

    # Συνάρτηση για εύρεση εγγράφων που περιέχουν έναν όρο
    def fetch_docs(term):
        return set(index_data.get(term, []))

    current_set = set()
    operation = None

    # Επεξεργασία του query
    for term in terms:
        if term.upper() in ["AND", "OR", "NOT"]:
            operation = term.upper()
        else:
            docs = fetch_docs(term)

            if operation == "AND":
                current_set &= docs
            elif operation == "OR":
                current_set |= docs
            elif operation == "NOT":
                current_set -= docs
            else:  # Πρώτος όρος στο query
                current_set = docs

            operation = None

    results = current_set
    return results

# Εκτέλεση του προγράμματος για queries
def main():
    print("Loading the inverted index...")
    index_data = load_index()
    print("Inverted index loaded successfully.")

    print("\nEnter your query (e.g., 'term1 AND term2 OR NOT term3'):")
    while True:
        query = input("\nQuery (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        matching_docs = boolean_retrieval(query, index_data)
        if matching_docs:
            print(f"\nDocuments matching your query:\n{matching_docs}")
        else:
            print("\nNo matching documents found.")

if __name__ == "__main__":
    main()
