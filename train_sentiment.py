"""Train the review-sentiment model used by the Flask application.

Run with the project virtual environment:
    .\\.venv\\Scripts\\python.exe train_sentiment.py
"""

from pathlib import Path
import pickle


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "reviews.txt"
MODEL_FILE = PROJECT_DIR / "nlp_model.pkl"
VECTORIZER_FILE = PROJECT_DIR / "tranform.pkl"  # Kept for app compatibility.


def load_reviews() -> tuple[pd.Series, pd.Series]:
    """Load and validate the labelled review dataset."""
    dataset = pd.read_csv(
        DATA_FILE,
        sep="\t",
        names=["label", "comment"],
        dtype={"label": "int64", "comment": "string"},
    ).dropna(subset=["comment"])

    if dataset.empty or not set(dataset["label"].unique()).issubset({0, 1}):
        raise ValueError("reviews.txt must contain non-empty reviews labelled 0 or 1.")

    return dataset["comment"], dataset["label"]


def main() -> None:
    comments, labels = load_reviews()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        comments,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # scikit-learn's built-in English list avoids an NLTK corpus download.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="ascii",
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    evaluation_model = MultinomialNB().fit(X_train, y_train)
    accuracy = accuracy_score(y_test, evaluation_model.predict(X_test))

    # Refit on every available review before writing the production artifacts.
    final_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="ascii",
        stop_words="english",
    )
    X_all = final_vectorizer.fit_transform(comments)
    final_model = MultinomialNB().fit(X_all, labels)

    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(final_model, model_file)
    with VECTORIZER_FILE.open("wb") as vectorizer_file:
        pickle.dump(final_vectorizer, vectorizer_file)

    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Trained on {len(comments)} reviews with {X_all.shape[1]} features.")
    print(f"Saved: {MODEL_FILE.name}, {VECTORIZER_FILE.name}")


if __name__ == "__main__":
    main()
