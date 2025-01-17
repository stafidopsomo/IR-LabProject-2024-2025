from search_engine import load_documents, load_inverted_index, boolean_search, tfidf_retrieval, bm25_retrieval

# Test queries
test_queries = [
    {"query": "maria", "expected": ["Main Page"]},
    {"query": "maria AND trubnikova", "expected": ["Main Page"]},
    {"query": "maria OR russian", "expected": ["Main Page", "Wikipedia"]},
    {"query": "russian AND NOT maria", "expected": ["Wikipedia"]},
]

# Automated tests for retrieval models
def test_boolean_search():
    print("\nTesting Boolean Retrieval...")
    titles, _ = load_documents("processed_articles.csv")
    inverted_index = load_inverted_index("inverted_index.json")

    for test in test_queries:
        query = test["query"]
        expected = set(test["expected"])
        result = boolean_search(query, inverted_index)
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        print(f"Pass: {result == expected}\n")

def test_tfidf():
    print("\nTesting TF-IDF Retrieval...")
    titles, documents = load_documents("processed_articles.csv")
    inverted_index = load_inverted_index("inverted_index.json")

    for test in test_queries:
        query = test["query"]
        expected = set(test["expected"])
        ranked_indices, scores = tfidf_retrieval(query, documents, inverted_index, titles)
        result = {titles[idx] for idx in ranked_indices}
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        print(f"Pass: {result == expected}\n")

def test_bm25():
    print("\nTesting Okapi BM25 Retrieval...")
    titles, documents = load_documents("processed_articles.csv")
    inverted_index = load_inverted_index("inverted_index.json")

    for test in test_queries:
        query = test["query"]
        expected = set(test["expected"])
        ranked_indices, scores = bm25_retrieval(query, documents, inverted_index, titles)
        result = {titles[idx] for idx in ranked_indices}
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Result: {result}")
        print(f"Pass: {result == expected}\n")

# Run all tests
if __name__ == "__main__":
    test_boolean_search()
    test_tfidf()
    test_bm25()
