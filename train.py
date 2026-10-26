
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "cleaned_resumes.csv"
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"

# Load cleaned data
df = pd.read_csv(DATA_PATH)

X = df["cleaned_resume"].fillna("")
y = df["Category"]

# Stratified split preserves category proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of categories:", y.nunique())

# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate on unseen test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("\nMODEL EVALUATION")
print("=" * 50)
print(f"Accuracy:   {accuracy:.4f}")
print(f"Macro-F1:   {macro_f1:.4f}")
print(f"Weighted-F1:{weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("\nClass label order:")
print(list(model.classes_))

# Save the entire fitted pipeline
joblib.dump(model, MODEL_PATH)
print("\nTrained model saved to:", MODEL_PATH)
