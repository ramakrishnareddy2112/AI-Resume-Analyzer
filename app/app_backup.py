import streamlit as st
import pdfplumber
from docx import Document
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f1b2d 0%, #05070d 45%, #020617 100%);
    color: white;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero-title {
    font-size: 58px;
    font-weight: 900;
    text-align: center;
    margin-top: 20px;
    letter-spacing: -1px;
}

.hero-title span {
    color: #22c55e;
}

.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 21px;
    margin-bottom: 35px;
}

.info-panel {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(14,165,233,0.08));
    border: 1.5px solid rgba(34,197,94,0.55);
    box-shadow: 0 0 22px rgba(34,197,94,0.20);
    border-radius: 24px;
    padding: 34px 35px;
    text-align: center;
    margin: 20px 0 18px 0;
}

.info-title {
    font-size: 34px;
    font-weight: 900;
    color: white;
    margin-bottom: 10px;
}

.info-subtitle {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 6px;
}

.info-note {
    color: #94a3b8;
    font-size: 16px;
}

.info-note b {
    color: #facc15;
}

.privacy-box {
    background: rgba(30,41,59,0.9);
    border: 1px solid rgba(148,163,184,0.25);
    padding: 16px 22px;
    border-radius: 14px;
    color: #cbd5e1;
    margin-bottom: 24px;
}

[data-testid="stFileUploader"] {
    background-color: rgba(30,41,59,0.98);
    border: 2px dashed #22c55e;
    border-radius: 18px;
    padding: 18px;
    margin-top: 8px;
    margin-bottom: 22px;
}

[data-testid="stFileUploader"] section {
    border: none;
}

textarea {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #ec4899 !important;
    border-radius: 14px !important;
}

.metric-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    color: white;
}

.skill-card {
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    padding: 10px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    font-weight: 800;
}

.matched-skill-card {
    background: linear-gradient(135deg,#16a34a,#22c55e);
    padding: 10px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    font-weight: 800;
}

.missing-skill-card {
    background: linear-gradient(135deg,#dc2626,#f97316);
    padding: 10px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
    font-weight: 800;
}

.section-box {
    background: #111827;
    border: 1px solid rgba(148,163,184,0.20);
    padding: 18px;
    border-radius: 15px;
    margin-top: 12px;
}

.feature-card {
    background: #111827;
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    transition: 0.25s;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: #22c55e;
    box-shadow: 0 0 20px rgba(34,197,94,0.22);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<div class="hero-title">🤖 AI Resume <span>Analyzer</span></div>
<div class="hero-subtitle">
Upload your resume and compare it with any job description using AI-powered ATS analysis.
</div>
""", unsafe_allow_html=True)

# ---------------- FILE EXTRACTORS ----------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return str(file.read(), "utf-8")

# ---------------- SKILLS DATABASE ----------------

skills_list = [
    "python", "sql", "machine learning", "deep learning", "excel", "power bi",
    "tableau", "pandas", "numpy", "tensorflow", "keras", "communication",
    "leadership", "java", "c++", "html", "css", "javascript", "react",
    "streamlit", "data analysis", "nlp", "statistics", "data visualization",
    "git", "github", "flask", "django", "mysql", "mongodb", "docker", "aws",
    "scikit-learn", "matplotlib", "seaborn", "plotly", "natural language processing",
    "computer vision", "data cleaning", "data preprocessing", "api", "rest api",
    "cloud", "linux", "powerpoint", "problem solving"
]

achievement_words = [
    "improved", "developed", "increased", "reduced", "built",
    "created", "managed", "designed", "achieved", "deployed",
    "implemented", "optimized", "automated", "analyzed"
]

# ---------------- ANALYSIS FUNCTIONS ----------------

def detect_experience(text):
    patterns = [r"(\d+)\+?\s+years", r"(\d+)\+?\s+months"]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group()
    return "Not Mentioned"

def count_achievements(text):
    return sum(text.lower().count(word) for word in achievement_words)

def extract_skills_from_text(text):
    detected = []
    text_lower = text.lower()
    for skill in skills_list:
        if skill.lower() in text_lower:
            detected.append(skill)
    return sorted(list(set(detected)))

def get_matched_and_missing_skills(resume_text, job_description):
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(job_description)

    matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing_skills = sorted(list(set(jd_skills).difference(set(resume_skills))))

    return resume_skills, jd_skills, matched_skills, missing_skills

def calculate_ats_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0.0, 0.0, 0.0

    vectorizer = CountVectorizer().fit_transform([resume_text, jd_text])
    keyword_similarity = cosine_similarity(vectorizer)[0][1]
    keyword_score = float(keyword_similarity) * 40

    embeddings = model.encode([resume_text, jd_text])
    semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    semantic_score = float(semantic_similarity) * 40

    formatting_score = 10
    achievement_score = min(count_achievements(resume_text) * 2, 10)

    final_score = keyword_score + semantic_score + formatting_score + achievement_score
    final_score = round(min(float(final_score), 100.0), 2)

    return final_score, round(keyword_score, 2), round(semantic_score, 2)

# ---------------- UPLOAD INFO PANEL ----------------

st.markdown("""
<div class="info-panel">
    <div style="font-size:52px;">📄</div>
    <div class="info-title">Resume File Upload</div>
    <div class="info-subtitle">
        Use the upload control directly below this panel.
    </div>
    <div class="info-note">
        Accepted formats: <b>PDF, DOCX, TXT</b>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your resume file here",
    type=["pdf", "docx", "txt"]
)

st.markdown("""
<div class="privacy-box">
ℹ️ Max file size: <b>200MB</b> per file • Your resume is processed only for this app session.
</div>
""", unsafe_allow_html=True)

# ---------------- JOB DESCRIPTION ----------------

st.markdown("## 💼 Paste Job Description")

job_description = st.text_area(
    "",
    height=220,
    placeholder="Paste company job description here..."
)

# ---------------- PROCESS ----------------

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    resume_text = ""

    try:
        if file_type == "pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            resume_text = extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            resume_text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file format.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

    if resume_text:
        st.success("✅ Resume Uploaded Successfully")

        resume_skills, jd_skills, matched_skills, missing_skills = get_matched_and_missing_skills(
            resume_text,
            job_description
        )

        experience = detect_experience(resume_text)
        match_score, keyword_score, semantic_score = calculate_ats_score(resume_text, job_description)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🎯 ATS Score</h2>
                <h1>{match_score}%</h1>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(float(match_score) / 100, 1.0))

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>💼 Experience</h2>
                <h1>{experience}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>🧠 Resume Skills</h2>
                <h1>{len(resume_skills)}</h1>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- SCORE BREAKDOWN ----------------

        st.markdown("## 📊 ATS Score Breakdown")

        b1, b2, b3 = st.columns(3)

        with b1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Keyword Match</h3>
                <h2>{keyword_score}/40</h2>
            </div>
            """, unsafe_allow_html=True)

        with b2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Semantic AI Match</h3>
                <h2>{semantic_score}/40</h2>
            </div>
            """, unsafe_allow_html=True)

        with b3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Structure + Impact</h3>
                <h2>20/20 max</h2>
            </div>
            """, unsafe_allow_html=True)

        if not job_description.strip():
            st.warning("Paste a job description to calculate a real ATS match score.")

        # ---------------- RESUME SKILLS ----------------

        st.markdown("## 🚀 Extracted Resume Skills")

        if resume_skills:
            cols = st.columns(4)
            for i, skill in enumerate(resume_skills):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="skill-card">
                        {skill.upper()}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No resume skills detected from current skill database.")

        # ---------------- MATCHED SKILLS ----------------

        if job_description.strip():
            st.markdown("## ✅ Matched Skills with Job Description")

            if matched_skills:
                cols = st.columns(4)
                for i, skill in enumerate(matched_skills):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div class="matched-skill-card">
                            {skill.upper()}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No matched skills detected with the job description.")

            # ---------------- MISSING SKILLS ----------------

            st.markdown("## ❌ Missing Skills from Resume")

            if missing_skills:
                cols = st.columns(4)
                for i, skill in enumerate(missing_skills):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div class="missing-skill-card">
                            {skill.upper()}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No important skills missing from the current skills database.")

        # ---------------- AI SUGGESTIONS ----------------

        st.markdown("## 🧠 AI Suggestions")

        suggestions = []

        if not job_description.strip():
            suggestions.append("Paste a job description to generate a more accurate ATS match score.")

        if match_score < 50 and job_description.strip():
            suggestions.append("Improve role alignment by adding more relevant skills and keywords from the job description.")

        if len(resume_skills) < 5:
            suggestions.append("Add more technical skills to improve ATS ranking.")

        if job_description.strip() and len(missing_skills) > 0:
            suggestions.append("Add missing skills if you genuinely know them: " + ", ".join(missing_skills[:6]) + ".")

        if count_achievements(resume_text) < 3:
            suggestions.append("Include measurable achievements using action verbs like developed, optimized, deployed, improved.")

        if "project" not in resume_text.lower():
            suggestions.append("Add a clear project section with tools, methods, and outcomes.")

        if "education" not in resume_text.lower():
            suggestions.append("Add education details clearly.")

        if "github" not in resume_text.lower():
            suggestions.append("Add your GitHub profile link.")

        if "linkedin" not in resume_text.lower():
            suggestions.append("Add your LinkedIn profile link.")

        if len(suggestions) == 0:
            suggestions.append("Excellent resume. Your ATS profile looks strong.")

        for suggestion in suggestions:
            st.markdown(f"""
            <div class="section-box">
                ✅ {suggestion}
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📄 Resume Text Preview"):
            st.write(resume_text[:5000])

else:
    st.info("Upload a resume to start analysis.")

# ---------------- FEATURE CARDS ----------------

st.markdown("## 🔥 Platform Capabilities")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <h2>📊 ATS Score</h2>
        <p>AI-powered score based on keywords, semantic match, achievements, and formatting.</p>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <h2>🧠 Semantic AI</h2>
        <p>Transformer-based NLP compares resume meaning with job descriptions.</p>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <h2>💼 Job Matching</h2>
        <p>Identify resume-job compatibility, matched skills, and missing skills.</p>
    </div>
    """, unsafe_allow_html=True)