
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "archive (2)" / "Resume" / "Resume.csv"
OUTPUT_PATH = BASE_DIR / "cleaned_resumes.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Keep only the columns required for classification
df = df[["ID", "Resume_str", "Category"]].copy()

# Clean resume text
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# Remove empty resumes
empty_count = (df["cleaned_resume"].str.len() == 0).sum()
df = df[df["cleaned_resume"].str.len() > 0].copy()

# Remove duplicate resume texts
duplicate_count = df["cleaned_resume"].duplicated().sum()
df = df.drop_duplicates(subset="cleaned_resume").copy()

# Remove rows with missing target labels
df = df.dropna(subset=["Category"])

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nPREPROCESSING RESULTS")
print("-" * 40)
print("Empty resumes removed:", empty_count)
print("Duplicate resumes removed:", duplicate_count)
print("Cleaned dataset shape:", df.shape)
print("Number of categories:", df["Category"].nunique())

print("\nCategory counts:")
print(df["Category"].value_counts().to_string())

print("\nSaved cleaned dataset to:", OUTPUT_PATH)
