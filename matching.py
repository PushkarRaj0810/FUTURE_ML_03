
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = [
    "python", "java", "c++", "c", "sql", "machine learning",
    "deep learning", "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "nlp",
    "natural language processing", "html", "css", "javascript",
    "react", "node.js", "django", "flask", "fastapi",
    "aws", "azure", "docker", "kubernetes", "git",
    "excel", "power bi", "tableau", "statistics",
    "data visualization", "communication", "project management",
    "leadership", "recruitment", "human resources",
    "financial analysis", "accounting", "rest api",
    "linux", "mongodb", "postgresql", "mysql",
]


def clean_text(text):
    """Normalize text for similarity matching."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text):
    """Extract skills from text using a predefined skill vocabulary."""
    text = clean_text(text)
    found = set()

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(skill)

    return found


def match_candidates(job_description, resumes, required_skills=None):
    """
    Rank resumes using TF-IDF similarity and required-skill coverage.

    resumes: list of dictionaries containing candidate and resume_text.
    required_skills: list of required skills; if omitted, extract skills
    from the job description.
    """
    job_description = clean_text(job_description)

    if not job_description:
        raise ValueError("Job description cannot be empty.")

    if not resumes:
        raise ValueError("Provide at least one resume.")

    resume_texts = [
        clean_text(resume.get("resume_text", ""))
        for resume in resumes
    ]

    if any(not text for text in resume_texts):
        raise ValueError("Every resume must contain readable text.")

    # Fit only on the job description and candidate resumes.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        [job_description] + resume_texts
    )

    similarities = cosine_similarity(
        matrix[0:1], matrix[1:]
    ).ravel()

    
    if required_skills is None:
        required = extract_skills(job_description)
    else:
        required = {
            clean_text(skill)
            for skill in required_skills
            if clean_text(skill)
        }

    results = []

    for index, resume in enumerate(resumes):
        resume_text = resume_texts[index]
        matched = []
        missing = []

        for skill in sorted(required):
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

            if re.search(pattern, resume_text, flags=re.IGNORECASE):
                matched.append(skill)
            else:
                missing.append(skill)

        coverage = (
            len(matched) / len(required) * 100
            if required else None
        )

        similarity_score = round(
            float(similarities[index]) * 100, 2
        )

        final_score = (
            round(0.60 * similarity_score + 0.40 * coverage, 2)
            if coverage is not None
            else similarity_score
        )

        results.append({
            "candidate": resume.get(
                "candidate", f"Candidate {index + 1}"
            ),
            "similarity_score": similarity_score,
            "skill_coverage": (
                round(coverage, 2) if coverage is not None else None
            ),
            "final_score": final_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "predicted_category": None,
        })

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return results
