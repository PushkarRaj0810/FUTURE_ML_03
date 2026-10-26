
# FUTURE_ML_03 — AI Resume Screening System

An ML-powered resume screening application that predicts resume categories, compares resumes with job descriptions, identifies matching and missing skills, and ranks candidates.

## Features

- PDF resume text extraction
- Resume category classification
- TF-IDF-based job-description similarity
- Required-skill coverage and skill-gap analysis
- Combined candidate ranking score
- Candidate results export to CSV
- Interactive Streamlit interface

## Machine Learning

The resume classifier uses TF-IDF word and character features with a Linear Support Vector Machine (Linear SVM).

The best evaluated model achieved:
- Test accuracy: approximately 66%
- Test macro-F1: 0.6194
- Test weighted-F1: 0.6485

The dataset contains 2,481 cleaned resumes across 24 job categories.

## Ranking Method

When required skills are supplied:

Final Score = 0.60 × Similarity Score + 0.40 × Skill Coverage

These weights are heuristic and have not been validated against actual hiring outcomes. Scores are not hiring probabilities.

## Installation

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Place the original dataset at the expected local path before running preprocessing or training scripts.

## Run the Application

```bash
streamlit run app.py
```

Upload text-based PDF resumes, enter a job description and required skills, and view the candidate rankings.

## Limitations

- Scanned PDFs require OCR, which is not currently implemented.
- Skill extraction uses a predefined vocabulary or explicitly entered required skills.
- The category classifier has limited performance on some underrepresented categories.
- Ranking scores support human review and should not be used as the sole basis for hiring decisions.
