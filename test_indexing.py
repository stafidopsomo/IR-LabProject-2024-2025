import pandas as pd
df = pd.read_csv("processed_articles.csv")
df['stemmed_tokens'] = df['stemmed_tokens'].apply(eval)  # Parse lists
print(df[df['stemmed_tokens'].apply(lambda tokens: 'january' in tokens)])
