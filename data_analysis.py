
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).parent
    / "archive (2)"
    / "Resume"
    / "Resume.csv"
)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 3 rows:\n", df[["ID", "Category", "Resume_str"]].head(3))

# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate resume IDs:", df["ID"].duplicated().sum())
print(
    "Duplicate resume texts:",
    df["Resume_str"].dropna().duplicated().sum()
)

# Category distribution
print("\nNumber of categories:", df["Category"].nunique())
print("\nCategory distribution:\n")
print(df["Category"].value_counts().to_string())

# Resume text length
text_lengths = df["Resume_str"].fillna("").str.split().str.len()

print("\nResume word-count statistics:")
print(text_lengths.describe())

print("\nShortest resume word count:", text_lengths.min())
print("Longest resume word count:", text_lengths.max())

print("\nDataset analysis completed.")
