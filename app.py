
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from pypdf import PdfReader

from matching import match_candidates


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "resume_classifier_best.joblib"


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def predict_category(model, resume_text):
    return str(model.predict([resume_text])[0])


st.title("AI-Powered Resume Screening System")
st.write(
    "Rank resumes against a job description, identify skill gaps, "
    "and predict each resume's likely job category."
)

st.info(
    "Similarity and skill coverage are matching indicators, "
    "not probabilities of being hired."
)

# Job requirements
st.header("1. Job Requirements")

job_description = st.text_area(
    "Paste the job description",
    height=200,
    placeholder=(
        "Example: We need a Python developer with SQL, "
        "machine learning, pandas, and Git experience."
    ),
)

required_skills_text = st.text_input(
    "Required skills, separated by commas (optional)",
    placeholder="Python, SQL, machine learning, pandas, Git",
    help=(
        "If left blank, the system extracts skills from the job "
        "description using its built-in skill vocabulary."
    ),
)

# Resume uploads
st.header("2. Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more resumes in PDF format",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Screen and Rank Candidates", type="primary"):
    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        st.stop()

    # Read the uploaded resumes
    candidates = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            resume_text = extract_pdf_text(uploaded_file.getvalue())

            if not resume_text.strip():
                errors.append(
                    f"{uploaded_file.name}: no readable text found."
                )
                continue

            candidates.append({
                "candidate": Path(uploaded_file.name).stem,
                "resume_text": resume_text,
            })

        except Exception as exc:
            errors.append(
                f"{uploaded_file.name}: could not read PDF ({exc})."
            )

    for error in errors:
        st.warning(error)

    if not candidates:
        st.error(
            "No readable resumes were found. Check that the PDFs "
            "contain selectable text rather than scanned images."
        )
        st.stop()

    # Parse explicitly supplied skills
    required_skills = None

    if required_skills_text.strip():
        required_skills = [
            skill.strip()
            for skill in required_skills_text.split(",")
            if skill.strip()
        ]

    # Match candidates against the job description
    try:
        with st.spinner("Analyzing resumes..."):
            results = match_candidates(
                job_description=job_description,
                resumes=candidates,
                required_skills=required_skills,
            )

            model = load_model()

            # Predict category independently of job-fit scores
            category_predictions = {
                candidate["candidate"]: predict_category(
                    model, candidate["resume_text"]
                )
                for candidate in candidates
            }

            for result in results:
                result["predicted_category"] = category_predictions[
                    result["candidate"]
                ]

    except Exception as exc:
        st.error(f"Screening failed: {exc}")
        st.stop()

    # Summary
    st.header("3. Screening Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Resumes processed", len(results))
    col2.metric(
    "Highest similarity score",
    f'{max(r["similarity_score"] for r in results):.2f}%'
    )

    coverage_values = [
        result["skill_coverage"]
        for result in results
        if result["skill_coverage"] is not None
    ]

    col3.metric(
        "Top candidate skill coverage",
        (
            f'{results[0]["skill_coverage"]:.2f}%'
            if results[0]["skill_coverage"] is not None
            else "Not available"
        ),
    )

    # Ranking table
    table_rows = []

    for rank, result in enumerate(results, start=1):
        table_rows.append({
            "Rank": rank,
            "Candidate": result["candidate"],
            "Similarity (%)": result["similarity_score"],
            "Final Score (%)": result["final_score"],
            "Skill Coverage (%)": result["skill_coverage"],
            "Predicted Category": result["predicted_category"],
            "Matched Skills": ", ".join(result["matched_skills"]),
            "Missing Skills": ", ".join(result["missing_skills"]),
        })

    results_df = pd.DataFrame(table_rows)

    st.subheader("Candidate Ranking")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results as CSV",
        data=csv_data,
        file_name="candidate_screening_results.csv",
        mime="text/csv",
    )

    # Detailed candidate reports
    st.subheader("Candidate Details")

    for rank, result in enumerate(results, start=1):
        with st.expander(
            f'{rank}. {result["candidate"]} — '
            f'{result["similarity_score"]:.2f}% similarity'
        ):
            st.write(
                "**Predicted job category:**",
                result["predicted_category"],
            )
            st.write(
                "**Job-description similarity:**",
                f'{result["similarity_score"]:.2f}%',
            )

            if result["skill_coverage"] is not None:
                st.write(
                    "**Required-skill coverage:**",
                    f'{result["skill_coverage"]:.2f}%',
                )
            else:
                st.write(
                    "No required skills were identified, "
                    "so skill coverage could not be calculated."
                )

            st.write("**Matched skills**")
            if result["matched_skills"]:
                st.write(", ".join(result["matched_skills"]))
            else:
                st.write("No skills from the configured vocabulary matched.")

            st.write("**Missing skills**")
            if result["missing_skills"]:
                st.write(", ".join(result["missing_skills"]))
            else:
                st.write("No missing skills from the specified requirements.")
