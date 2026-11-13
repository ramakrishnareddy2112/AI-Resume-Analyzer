import streamlit as st
import pdfplumber
from docx import Document
import re
import os
from xml.sax.saxutils import escape
from google import genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_option_menu import option_menu
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

default_values = {
    "uploader_key": 0,
    "resume_uploaded": False,
    "resume_file_name": "",
    "resume_text": "",
    "job_description": "",
    "resume_skills": [],
    "jd_skills": [],
    "matched_skills": [],
    "missing_skills": [],
    "match_score": 0.0,
    "keyword_score": 0.0,
    "semantic_score": 0.0,
    "formatting_score": 0.0,
    "achievement_score": 0.0,
    "gemini_feedback": "",
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_current_resume():
    keys_to_clear = [
        "resume_uploaded", "resume_file_name", "resume_text", "job_description",
        "resume_skills", "jd_skills", "matched_skills", "missing_skills",
        "match_score", "keyword_score", "semantic_score", "formatting_score",
        "achievement_score", "gemini_feedback",
    ]

    for key in keys_to_clear:
        st.session_state[key] = default_values[key]

    st.session_state["uploader_key"] += 1


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ---------------------------------------------------------------------------
# Gemini (google-genai SDK)
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_api_key():
    """Read the key from Streamlit secrets, falling back to an environment variable."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def get_gemini_model_name():
    """Model name can be overridden with GEMINI_MODEL in Streamlit secrets."""
    try:
        return st.secrets["GEMINI_MODEL"]
    except Exception:
        return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def generate_gemini_feedback(resume_text, job_description, missing_skills, match_score):
    api_key = get_gemini_api_key()

    if not api_key:
        return "Gemini API key not found. Add GEMINI_API_KEY in Streamlit Secrets (Manage app > Settings > Secrets)."

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a professional ATS resume reviewer and career coach.

Analyze this resume for the given job description.

ATS Score: {match_score}%

Missing Skills:
{", ".join(missing_skills) if missing_skills else "No major missing skills detected"}

Resume Text:
{resume_text[:5000]}

Job Description:
{job_description[:3000]}

Give output strictly in this format:

1. Overall Resume Review
2. Strengths
3. Weaknesses
4. Missing Keywords
5. ATS Optimization Suggestions
6. Improved Professional Summary
7. Improved Project Bullet Points
8. Final Recruiter Advice

Keep it practical, professional, and useful for a student applying for internships.
"""

        response = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=prompt,
        )
        return response.text or "Gemini returned an empty response. Please try again."

    except Exception as e:
        return f"Gemini AI error: {e}"


def parse_gemini_sections(text):
    sections = {
        "Overall Resume Review": "",
        "Strengths": "",
        "Weaknesses": "",
        "Missing Keywords": "",
        "ATS Optimization Suggestions": "",
        "Improved Professional Summary": "",
        "Improved Project Bullet Points": "",
        "Final Recruiter Advice": "",
    }

    headings = list(sections.keys())
    current_heading = None

    for line in text.splitlines():
        clean_line = line.strip()
        matched_heading = None

        for heading in headings:
            pattern = heading.lower()
            if pattern in clean_line.lower():
                matched_heading = heading
                break

        if matched_heading:
            current_heading = matched_heading
            clean_line = re.sub(r"^\d+\.\s*", "", clean_line)
            clean_line = clean_line.replace(matched_heading, "").strip(" :-")
            if clean_line:
                sections[current_heading] += clean_line + "\n"
        else:
            if current_heading and clean_line:
                sections[current_heading] += clean_line + "\n"

    return sections


def render_gemini_cards(feedback_text):
    sections = parse_gemini_sections(feedback_text)

    icons = {
        "Overall Resume Review": "📋",
        "Strengths": "✅",
        "Weaknesses": "⚠️",
        "Missing Keywords": "🔍",
        "ATS Optimization Suggestions": "🎯",
        "Improved Professional Summary": "🧾",
        "Improved Project Bullet Points": "🚀",
        "Final Recruiter Advice": "🏢",
    }

    for title, content in sections.items():
        if content.strip():
            st.markdown(
                f"""
                <div class='gemini-card'>
                    <h3>{icons.get(title, "✨")} {title}</h3>
                    <p>{escape(content.strip())}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


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

.status-box {
    background: #111827;
    border: 1px solid rgba(34,197,94,0.45);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 18px;
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

.gemini-card {
    background: linear-gradient(135deg, #111827, #1e1b4b);
    border: 1px solid rgba(129,140,248,0.45);
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    padding: 22px;
    border-radius: 18px;
    margin-top: 18px;
    color: #e5e7eb;
    white-space: pre-wrap;
}

.gemini-card h3 {
    color: #a5b4fc;
    margin-bottom: 12px;
}

[data-testid="stFileUploader"] {
    background-color: rgba(30,41,59,0.98);
    border: 2px dashed #22c55e;
    border-radius: 18px;
    padding: 18px;
    margin-top: 8px;
    margin-bottom: 22px;
}

textarea {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #ec4899 !important;
    border-radius: 14px !important;
}

.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 12px;
    font-weight: 800;
    border: none;
}

.stButton>button:hover {
    background-color: #22c55e;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-title">🤖 AI Resume <span>Analyzer</span></div>
<div class="hero-subtitle">
Corporate-style resume screening with persistent analysis, ATS score, semantic AI matching, Gemini feedback, recruiter insights, and downloadable reports.
</div>
""", unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["Analyzer", "Dashboard", "Insights", "Gemini AI"],
    icons=["cloud-upload", "bar-chart", "lightbulb", "stars"],
    orientation="horizontal",
    styles={
        "container": {"padding": "8px", "background-color": "#111827"},
        "icon": {"color": "#22c55e", "font-size": "18px"},
        "nav-link": {"font-size": "17px", "text-align": "center", "color": "white"},
        "nav-link-selected": {"background-color": "#16a34a"},
    },
)


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


skill_aliases = {
    "python": ["python", "python programming"],
    "sql": ["sql", "mysql", "postgresql"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "tensorflow", "keras", "pytorch"],
    "data science": ["data science", "data scientist"],
    "data analysis": ["data analysis", "data analyst", "eda"],
    "data visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "ms excel"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "opencv"],
    "streamlit": ["streamlit"],
    "git": ["git"],
    "github": ["github"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "flask": ["flask"],
    "django": ["django"],
    "mongodb": ["mongodb", "mongo db"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "api": ["api", "rest api"],
    "linux": ["linux"],
    "communication": ["communication"],
    "leadership": ["leadership"],
    "problem solving": ["problem solving", "problem-solving"],
}

achievement_words = [
    "improved", "developed", "increased", "reduced", "built", "created",
    "managed", "designed", "achieved", "deployed", "implemented",
    "optimized", "automated", "analyzed"
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_phrase(text, phrase):
    phrase = phrase.lower().strip()

    if len(phrase) <= 3:
        pattern = r"(?<![a-zA-Z0-9+#])" + re.escape(phrase) + r"(?![a-zA-Z0-9+#])"
    else:
        pattern = r"\b" + re.escape(phrase) + r"\b"

    return re.search(pattern, text) is not None


def extract_skills_from_text(text):
    text_clean = clean_text(text)
    detected = set()

    for main_skill, aliases in skill_aliases.items():
        for alias in aliases:
            if contains_phrase(text_clean, alias):
                detected.add(main_skill)
                break

    return sorted(detected)


def detect_experience(text):
    patterns = [r"(\d+)\+?\s+years", r"(\d+)\+?\s+months"]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group()

    return "Not Mentioned"


def count_achievements(text):
    text_clean = clean_text(text)
    return sum(text_clean.count(word) for word in achievement_words)


def get_matched_and_missing_skills(resume_text, job_description):
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(job_description)

    matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing_skills = sorted(list(set(jd_skills).difference(set(resume_skills))))

    return resume_skills, jd_skills, matched_skills, missing_skills


def calculate_ats_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        vectorizer = CountVectorizer().fit_transform([resume_text, jd_text])
        keyword_similarity = cosine_similarity(vectorizer)[0][1]
        keyword_score = float(keyword_similarity) * 40
    except Exception:
        keyword_score = 10

    try:
        embeddings = model.encode([resume_text, jd_text])
        semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        semantic_score = float(semantic_similarity) * 40
    except Exception:
        semantic_score = 15

    formatting_score = 10
    achievement_score = min(count_achievements(resume_text) * 2, 10)

    final_score = keyword_score + semantic_score + formatting_score + achievement_score
    final_score = max(final_score, 35)
    final_score = round(min(float(final_score), 100.0), 2)

    return final_score, round(keyword_score, 2), round(semantic_score, 2), formatting_score, achievement_score


def recruiter_decision(score):
    if score >= 80:
        return "Highly Recommended", "#22c55e"
    elif score >= 60:
        return "Moderate Match", "#facc15"
    else:
        return "Needs Improvement", "#ef4444"


def generate_pdf_report():
    file_path = "ATS_Report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI Resume Analyzer Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    ats_score = st.session_state["match_score"]
    decision, _ = recruiter_decision(ats_score)

    sections = {
        "Resume File": st.session_state["resume_file_name"],
        "ATS Score": f"{ats_score}%",
        "Recruiter Decision": decision,
        "Keyword Match Score": str(st.session_state["keyword_score"]),
        "Semantic AI Match Score": str(st.session_state["semantic_score"]),
        "Formatting Score": str(st.session_state["formatting_score"]),
        "Achievement Score": str(st.session_state["achievement_score"]),
        "Resume Skills": ", ".join(st.session_state["resume_skills"]),
        "Matched Skills": ", ".join(st.session_state["matched_skills"]),
        "Missing Skills": ", ".join(st.session_state["missing_skills"]),
    }

    for heading, content in sections.items():
        elements.append(Paragraph(f"<b>{heading}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        # escape() prevents characters like & and < from breaking ReportLab's parser
        elements.append(Paragraph(escape(str(content)), styles["BodyText"]))
        elements.append(Spacer(1, 14))

    if st.session_state["gemini_feedback"]:
        elements.append(Paragraph("<b>Gemini AI Feedback</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        safe_text = escape(st.session_state["gemini_feedback"]).replace("\n", "<br/>")
        elements.append(Paragraph(safe_text, styles["BodyText"]))

    doc.build(elements)
    return file_path


def run_analysis(resume_text, job_description, file_name):
    resume_skills, jd_skills, matched_skills, missing_skills = get_matched_and_missing_skills(
        resume_text,
        job_description
    )

    match_score, keyword_score, semantic_score, formatting_score, achievement_score = calculate_ats_score(
        resume_text,
        job_description
    )

    st.session_state["resume_uploaded"] = True
    st.session_state["resume_file_name"] = file_name
    st.session_state["resume_text"] = resume_text
    st.session_state["job_description"] = job_description
    st.session_state["resume_skills"] = resume_skills
    st.session_state["jd_skills"] = jd_skills
    st.session_state["matched_skills"] = matched_skills
    st.session_state["missing_skills"] = missing_skills
    st.session_state["match_score"] = match_score
    st.session_state["keyword_score"] = keyword_score
    st.session_state["semantic_score"] = semantic_score
    st.session_state["formatting_score"] = formatting_score
    st.session_state["achievement_score"] = achievement_score
    st.session_state["gemini_feedback"] = ""


def show_analysis_results():
    resume_text = st.session_state["resume_text"]
    resume_skills = st.session_state["resume_skills"]
    matched_skills = st.session_state["matched_skills"]
    missing_skills = st.session_state["missing_skills"]
    match_score = st.session_state["match_score"]

    decision, decision_color = recruiter_decision(match_score)
    experience = detect_experience(resume_text)

    st.success("✅ Analysis completed and saved for this session")

    st.markdown(
        f"""
        <div class='status-box'>
            <b>Current Resume:</b> {escape(st.session_state["resume_file_name"])}<br>
            <b>Status:</b> Analysis available across Dashboard, Insights, and Gemini AI tabs
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>🎯 ATS Score</h3>
                <h1>{match_score}%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(min(float(match_score) / 100, 1.0))

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>💼 Experience</h3>
                <h1>{experience}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>🧠 Resume Skills</h3>
                <h1>{len(resume_skills)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3>🏢 Recruiter View</h3>
                <h2 style='color:{decision_color};'>{decision}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🚀 Extracted Resume Skills")

    if resume_skills:
        cols = st.columns(4)
        for i, skill in enumerate(resume_skills):
            with cols[i % 4]:
                st.markdown(f"<div class='skill-card'>{skill.upper()}</div>", unsafe_allow_html=True)

    if st.session_state["job_description"].strip():
        st.markdown("## ✅ Matched Skills")

        if matched_skills:
            cols = st.columns(4)
            for i, skill in enumerate(matched_skills):
                with cols[i % 4]:
                    st.markdown(f"<div class='matched-skill-card'>{skill.upper()}</div>", unsafe_allow_html=True)

        st.markdown("## ❌ Missing Skills")

        if missing_skills:
            cols = st.columns(4)
            for i, skill in enumerate(missing_skills):
                with cols[i % 4]:
                    st.markdown(f"<div class='missing-skill-card'>{skill.upper()}</div>", unsafe_allow_html=True)
        else:
            st.success("No major missing skills detected.")

    with st.expander("📄 Resume Text Preview"):
        st.write(resume_text[:5000])


if selected == "Analyzer":
    st.markdown("""
    <div class="info-panel">
        <div style="font-size:52px;">📄</div>
        <div style="font-size:34px; font-weight:900;">Resume File Upload</div>
        <div style="color:#cbd5e1; font-size:18px;">
            Upload once, analyze once, then move between tabs without losing results.
        </div>
        <div style="color:#94a3b8; font-size:16px;">
            Accepted formats: <b style="color:#facc15;">PDF, DOCX, TXT</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["resume_uploaded"]:
        st.markdown(
            f"""
            <div class='status-box'>
                <b>Current Resume Loaded:</b> {escape(st.session_state["resume_file_name"])}<br>
                You can view Dashboard, Insights, and Gemini AI without uploading again.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🗑️ Remove Current Resume and Start New"):
            clear_current_resume()
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload your resume file here",
        type=["pdf", "docx", "txt"],
        key=f"resume_uploader_{st.session_state['uploader_key']}"
    )

    st.markdown("## 💼 Paste Job Description")

    job_description = st.text_area(
        "Job description",
        height=220,
        placeholder="Paste company job description here, then click Analyze Resume below...",
        key="jd_input",
        label_visibility="collapsed"
    )

    analyze_button = st.button("🚀 Analyze Resume")

    if analyze_button:
        if uploaded_file is None and not st.session_state["resume_uploaded"]:
            st.error("Please upload a resume first.")
        elif not job_description.strip():
            st.error("Please paste a job description before analyzing.")
        else:
            resume_text = ""

            if uploaded_file is not None:
                file_type = uploaded_file.name.split(".")[-1].lower()

                try:
                    if file_type == "pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif file_type == "docx":
                        resume_text = extract_text_from_docx(uploaded_file)
                    elif file_type == "txt":
                        resume_text = extract_text_from_txt(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading file: {e}")

                file_name = uploaded_file.name
            else:
                resume_text = st.session_state["resume_text"]
                file_name = st.session_state["resume_file_name"]

            if resume_text:
                run_analysis(resume_text, job_description, file_name)
                st.rerun()

    if st.session_state["resume_uploaded"]:
        show_analysis_results()
    else:
        st.info("Upload a resume, paste job description, then click Analyze Resume.")


elif selected == "Dashboard":
    st.markdown("## 📊 Resume Analytics Dashboard")

    if not st.session_state["resume_uploaded"]:
        st.warning("Please analyze a resume first in the Analyzer tab.")
    else:
        match_score = st.session_state["match_score"]
        keyword_score = st.session_state["keyword_score"]
        semantic_score = st.session_state["semantic_score"]
        formatting_score = st.session_state["formatting_score"]
        achievement_score = st.session_state["achievement_score"]
        resume_skills = st.session_state["resume_skills"]
        matched_skills = st.session_state["matched_skills"]
        missing_skills = st.session_state["missing_skills"]

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_score,
            title={"text": "ATS Match Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e"},
                "steps": [
                    {"range": [0, 50], "color": "#7f1d1d"},
                    {"range": [50, 75], "color": "#713f12"},
                    {"range": [75, 100], "color": "#14532d"},
                ],
            }
        ))

        gauge.update_layout(paper_bgcolor="#020617", font={"color": "white"}, height=350)
        st.plotly_chart(gauge, use_container_width=True)

        chart_data = pd.DataFrame({
            "Category": ["Keyword Match", "Semantic AI Match", "Formatting", "Achievements"],
            "Score": [keyword_score, semantic_score, formatting_score, achievement_score]
        })

        fig_bar = px.bar(chart_data, x="Category", y="Score", text="Score", title="ATS Score Breakdown")
        fig_bar.update_layout(paper_bgcolor="#020617", plot_bgcolor="#020617", font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

        skill_counts = [len(resume_skills), len(matched_skills), len(missing_skills)]

        if sum(skill_counts) > 0:
            skill_chart = pd.DataFrame({
                "Type": ["Resume Skills", "Matched Skills", "Missing Skills"],
                "Count": skill_counts
            })

            fig_pie = px.pie(skill_chart, names="Type", values="Count", title="Skill Match Distribution")
            fig_pie.update_layout(paper_bgcolor="#020617", font_color="white")
            st.plotly_chart(fig_pie, use_container_width=True)


elif selected == "Insights":
    st.markdown("## 💡 Recruiter Insights")

    if not st.session_state["resume_uploaded"]:
        st.warning("Please analyze a resume first in the Analyzer tab.")
    else:
        score = st.session_state["match_score"]
        missing_skills = st.session_state["missing_skills"]
        matched_skills = st.session_state["matched_skills"]

        decision, color = recruiter_decision(score)

        st.markdown(
            f"""
            <div class='metric-card'>
                <h2>Recruiter Decision Prediction</h2>
                <h1 style='color:{color};'>{decision}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## Strengths")

        if matched_skills:
            st.success("Your resume matches important job skills: " + ", ".join(matched_skills[:8]))
        else:
            st.warning("No strong skill matches detected yet.")

        st.markdown("## Improvement Areas")

        if missing_skills:
            st.error("Missing important job skills: " + ", ".join(missing_skills[:8]))
        else:
            st.success("No major missing skills detected.")

        st.markdown("## Next Action Plan")

        if score >= 80:
            st.info("You can apply confidently. Still improve bullet points with measurable achievements.")
        elif score >= 60:
            st.info("Add missing keywords and strengthen project descriptions before applying.")
        else:
            st.info("Customize your resume strongly for this job role before applying.")


elif selected == "Gemini AI":
    st.markdown("## ✨ Gemini AI Resume Coach")

    if not st.session_state["resume_uploaded"]:
        st.warning("Please analyze a resume first in the Analyzer tab.")
    else:
        resume_text = st.session_state["resume_text"]
        job_description = st.session_state["job_description"]
        missing_skills = st.session_state["missing_skills"]
        match_score = st.session_state["match_score"]

        st.markdown(
            f"""
            <div class='status-box'>
                <b>Current Resume:</b> {escape(st.session_state["resume_file_name"])}<br>
                Gemini will use the saved resume and job description from your analysis.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Generate Gemini AI Feedback"):
            with st.spinner("Gemini is analyzing your resume..."):
                feedback = generate_gemini_feedback(
                    resume_text,
                    job_description,
                    missing_skills,
                    match_score
                )
                st.session_state["gemini_feedback"] = feedback

        if st.session_state["gemini_feedback"]:
            st.markdown("### Gemini AI Feedback")
            render_gemini_cards(st.session_state["gemini_feedback"])

            with st.expander("View Raw Gemini Output"):
                st.write(st.session_state["gemini_feedback"])

            pdf_file = generate_pdf_report()

            with open(pdf_file, "rb") as pdf:
                st.download_button(
                    label="📄 Download ATS PDF Report",
                    data=pdf,
                    file_name="ATS_Report.pdf",
                    mime="application/pdf"
                )