"""
Dimitrakopoulos Stylianos 
AM: 18390149
Προγραμμα Σπουδων ΠΑΔΑ
"""

from search_engine import load_documents, load_inverted_index, boolean_search, tfidf_retrieval, bm25_retrieval

# Φόρτωση αρχειων
titles, documents = load_documents("processed_articles.csv")
inverted_index = load_inverted_index("inverted_index.json")

# Test queries
test_queries = [
    {"query": "maria", "expected": {"Main Page"}},
    {"query": "maria AND trubnikova", "expected": {"Main Page"}},
    {"query": "maria OR russian", "expected": {"Main Page", "Wikipedia"}},
    {"query": "russian AND NOT maria", "expected": {"Wikipedia"}},
    {"query": "NOT maria", "expected": set()},  # Expect empty results
    {"query": "maria AND NOT russian", "expected": set()},  # Expect empty if "Main Page" contains "russian"
]

# Function για test όλων των retrieval methods
def run_tests(retrieval_function, method_name):
    print(f"\n🔍 Testing {method_name} Retrieval...")
    
    for test in test_queries:
        query = test["query"]
        expected = test["expected"]
        
        print(f"\n[DEBUG] Running {method_name} Search for query: {query}")

        if method_name == "Boolean":
            result = retrieval_function(query, inverted_index)
        else:
            ranked_indices, scores = retrieval_function(query, documents, inverted_index, titles)
            result = {titles[idx] for idx in ranked_indices} if ranked_indices else set()
        
        print(f"🔎 Expected: {expected}")
        print(f"✅ {method_name} Search Result: {result}")
        print(f"🟢 Pass: {result == expected}\n")

# Function for custom user queries
def run_custom_query():
    print("\nSelect a retrieval model:")
    print("1. Boolean Retrieval")
    print("2. TF-IDF Retrieval")
    print("3. Okapi BM25 Retrieval")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    model_mapping = {"1": ("Boolean", boolean_search), "2": ("TF-IDF", tfidf_retrieval), "3": ("Okapi BM25", bm25_retrieval)}
    
    if choice not in model_mapping:
        print("Invalid choice. Exiting...")
        return

    model_name, model_function = model_mapping[choice]
    query = input("\nEnter your query: ").strip()
    
    print(f"\n🔍 Running {model_name} Retrieval for query: {query}")

    if model_name == "Boolean":
        result = model_function(query, inverted_index)
    else:
        ranked_indices, scores = model_function(query, documents, inverted_index, titles)
        result = {titles[idx] for idx in ranked_indices} if ranked_indices else set()

    print(f"🔎 {model_name} Search Result: {result}\n")

# Main execution flow
if __name__ == "__main__":
    print("Would you like to run the predefined test queries or enter a custom query?")
    print("1. Run predefined test queries")
    print("2. Enter custom query")

    user_choice = input("Enter your choice (1/2): ").strip()

    if user_choice == "1":
        run_tests(boolean_search, "Boolean")
        run_tests(tfidf_retrieval, "TF-IDF")
        run_tests(bm25_retrieval, "Okapi BM25")
    elif user_choice == "2":
        run_custom_query()
    else:
        print("Invalid choice. Exiting...")
