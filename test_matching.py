
from matching import match_candidates

job_description = """
We need a Python developer with SQL, machine learning,
pandas, and Git experience.
"""

resumes = [
    {
        "candidate": "Candidate A",
        "resume_text": """
        Python developer with SQL, pandas, machine learning,
        Git, and data analysis experience.
        """,
    },
    {
        "candidate": "Candidate B",
        "resume_text": """
        Graphic designer skilled in Photoshop, illustration,
        branding, and visual communication.
        """,
    },
]

results = match_candidates(
    job_description,
    resumes,
    required_skills=["python", "sql", "machine learning", "pandas", "git"],
)

for result in results:
    print("\nCandidate:", result["candidate"])
    print("Similarity:", result["similarity_score"], "%")
    print("Skill coverage:", result["skill_coverage"], "%")
    print("Matched skills:", result["matched_skills"])
    print("Missing skills:", result["missing_skills"])

assert results[0]["candidate"] == "Candidate A"
assert results[0]["skill_coverage"] == 100.0
assert results[1]["skill_coverage"] == 0.0

print("\nAll matching tests passed.")
