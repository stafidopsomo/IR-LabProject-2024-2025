import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from search_engine import load_documents, load_inverted_index, boolean_search, tfidf_retrieval, bm25_retrieval

# Load necessary data
titles, documents = load_documents("processed_articles.csv")
inverted_index = load_inverted_index("inverted_index.json")

# Define ground truth for evaluation
ground_truth = {
    "maria": {"Main Page"},
    "maria AND trubnikova": {"Main Page"},
    "maria OR russian": {"Main Page", "Wikipedia"},
    "russian AND NOT maria": {"Wikipedia"},
    "NOT maria": set(),
    "maria AND NOT russian": set()
}

# Function to compute evaluation metrics
def evaluate_retrieval(retrieval_function, method_name, is_boolean=False):
    print(f"\n🔍 Evaluating {method_name} Retrieval...")

    precision_scores, recall_scores, f1_scores = [], [], []

    for query, expected in ground_truth.items():
        print(f"\n[DEBUG] Running {method_name} for query: {query}")

        # Boolean retrieval returns a set, TF-IDF and BM25 return (ranked_indices, scores)
        if is_boolean:
            result = retrieval_function(query, inverted_index)
        else:
            ranked_indices, _ = retrieval_function(query, documents, inverted_index, titles)
            result = {titles[idx] for idx in ranked_indices} if ranked_indices else set()

        # Convert results and ground truth to binary format
        all_documents = set(titles)
        y_true = [1 if doc in expected else 0 for doc in all_documents]
        y_pred = [1 if doc in result else 0 for doc in all_documents]

        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"🔎 Expected: {expected}")
        print(f"✅ {method_name} Search Result: {result}")
        print(f"🎯 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Compute average scores
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print(f"\n📊 {method_name} Evaluation Summary:")
    print(f"⚡ Average Precision: {avg_precision:.4f}")
    print(f"📈 Average Recall: {avg_recall:.4f}")
    print(f"📊 Average F1-score: {avg_f1:.4f}\n")

# Run evaluations
if __name__ == "__main__":
    evaluate_retrieval(boolean_search, "Boolean", is_boolean=True)
    evaluate_retrieval(tfidf_retrieval, "TF-IDF")
    evaluate_retrieval(bm25_retrieval, "Okapi BM25")
