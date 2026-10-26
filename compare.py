
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "cleaned_resumes.csv"

df = pd.read_csv(DATA_PATH)
X = df["cleaned_resume"].fillna("")
y = df["Category"]

# Reserve a final test set; use validation for model selection.
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.20,
    stratify=y_train_all,
    random_state=42,
)

def make_word_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )

def make_features():
    return FeatureUnion([
        ("word", make_word_vectorizer()),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
    ])

models = {
    "Logistic Regression (word TF-IDF)": Pipeline([
        ("tfidf", make_word_vectorizer()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
    "Linear SVM (word TF-IDF)": Pipeline([
        ("tfidf", make_word_vectorizer()),
        ("classifier", LinearSVC(class_weight="balanced")),
    ]),
    "Linear SVM (word + character TF-IDF)": Pipeline([
        ("tfidf", make_features()),
        ("classifier", LinearSVC(class_weight="balanced")),
    ]),
}

results = []

print("Training and comparing models...\n")

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted", zero_division=0
    )

    results.append({
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Weighted-F1: {weighted_f1:.4f}\n"
    )

# Choose using validation macro-F1
results.sort(key=lambda r: r["macro_f1"], reverse=True)
best = results[0]

print("=" * 60)
print("VALIDATION COMPARISON")
print("=" * 60)

print(pd.DataFrame([
    {
        "Model": r["name"],
        "Accuracy": round(r["accuracy"], 4),
        "Macro-F1": round(r["macro_f1"], 4),
        "Weighted-F1": round(r["weighted_f1"], 4),
    }
    for r in results
]).to_string(index=False))

# Evaluate the selected configuration on the held-out test set once.
test_predictions = best["model"].predict(X_test)

print("\nSELECTED MODEL:", best["name"])
print("Held-out test accuracy:",
      round(accuracy_score(y_test, test_predictions), 4))
print("Held-out test macro-F1:",
      round(f1_score(y_test, test_predictions,
                     average="macro", zero_division=0), 4))
print("Held-out test weighted-F1:",
      round(f1_score(y_test, test_predictions,
                     average="weighted", zero_division=0), 4))

print("\nTest classification report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# Retrain the selected configuration on all available labeled data.
# The test metrics above come from the model before this refit.
best["model"].fit(X, y)

output_path = BASE_DIR / "resume_classifier_best.joblib"
joblib.dump(best["model"], output_path)

print("\nFinal pipeline retrained on all cleaned data.")
print("Saved to:", output_path)
