from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pandas as pd

# Sample data: texts and their corresponding labels (assuming categorical labels)
texts = [
    "The movie was fantastic and thrilling",
    "A boring and slow-paced storyline",
    "Incredible performance by the lead actor",
    "The plot was uninteresting",
    "Great cinematography and music"
]
labels = [1, 0, 1, 0, 1]  # Example labels: 1 for positive, 0 for negative

# 1. Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)

# 2. Calculate the chi-square correlation between each feature and the labels
chi2_scores, p_values = chi2(X_tfidf, labels)

# Map chi-square scores to feature names (words)
feature_scores = pd.DataFrame(
    {'feature': vectorizer.get_feature_names_out(), 'chi2_score': chi2_scores, 'p_value': p_values}
)

# Sort features by their chi-square scores to see which words are most associated with the labels
feature_scores = feature_scores.sort_values(by='chi2_score', ascending=False)

print("Top words associated with the labels:")
print(feature_scores.head(10))

