"""Sparse text representations used by the clustering workflow."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_text(text, min_df=2, max_df=0.95, max_features=30000):
    """Return TF-IDF, Count BoW, and their fitted vectorizers."""
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=min_df,
        max_df=max_df, max_features=max_features, sublinear_tf=True,
    )
    bow_vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 1), min_df=min_df,
        max_df=max_df, max_features=max_features,
    )
    tfidf = tfidf_vectorizer.fit_transform(text)
    bow = bow_vectorizer.fit_transform(text)
    return tfidf, tfidf_vectorizer, bow, bow_vectorizer
