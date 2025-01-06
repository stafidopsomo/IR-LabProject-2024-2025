import json

# Load the inverted index
def load_inverted_index(file_path="inverted_index.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Perform Boolean operations
def boolean_search(query, inverted_index):
    terms = query.split()
    result_docs = set()

    # Helper to fetch documents for a term
    def get_docs(term):
        return set(inverted_index.get(term, []))

    # Process the query
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
            else:  # First term
                current_docs = docs

            operation = None

    result_docs = current_docs
    return result_docs

# Main function to handle queries
def main():
    print("Loading the inverted index...")
    inverted_index = load_inverted_index()
    print("Inverted index loaded successfully!")

    print("\nEnter your query (e.g., 'term1 AND term2 OR NOT term3'):")
    while True:
        query = input("\nQuery (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        results = boolean_search(query, inverted_index)
        if results:
            print(f"\nDocuments matching your query:\n{results}")
        else:
            print("\nNo matching documents found.")

# Run the query handler
if __name__ == "__main__":
    main()
