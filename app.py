import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Content Safety Analyzer", page_icon="🛡️", layout="wide")

# Streamlit Community Cloud secrets support
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    pass

st.markdown("""
<style>
.block-container {max-width:1180px; padding-top:5rem; padding-bottom:3rem;}
.hero {padding:1.4rem 1.5rem; border:1px solid rgba(128,128,128,.25); border-radius:18px; margin-bottom:1.2rem;}
.kicker {text-transform:uppercase; letter-spacing:.12em; font-size:.76rem; opacity:.7; font-weight:700;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:1rem; min-height:120px;}
.metric-label {font-size:.86rem; opacity:.75; margin-bottom:.35rem;}
.metric-score {font-size:2rem; font-weight:750; line-height:1.2;}
.metric-note {font-size:.8rem; opacity:.7; margin-top:.35rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="kicker">AI content review · LangGraph + Groq</div>
<h1>Content Safety Analyzer</h1>
<p>Analyze text across toxicity, copyright risk, and cultural sensitivity in one run.</p>
</div>
""", unsafe_allow_html=True)

SAMPLE_TEXT = (
    "A creator uses several paragraphs copied directly from a popular author's book "
    "without permission or attribution. The video mocks a religious community's traditional "
    "clothing and calls its customs backward. The creator then insults members of that "
    "community and says they do not belong in modern society."
)

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

def load_sample():
    st.session_state["input_text"] = SAMPLE_TEXT

def clear_input():
    st.session_state["input_text"] = ""

with st.sidebar:
    st.subheader("Analysis branches")
    st.markdown("- Toxicity & hate speech\n- Copyright risk\n- Cultural sensitivity")
    st.caption("Scores are model estimates, not legal advice or definitive moderation decisions.")

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.subheader("Text to analyze")
    st.caption("Paste a script, caption, comment, or transcript.")
    st.text_area(
        "Input text",
        key="input_text",
        height=280,
        placeholder="Paste your content here...",
        label_visibility="collapsed",
    )
    b1, b2, b3 = st.columns([1, 1, 1.25])
    with b1:
        st.button("Load Sample", on_click=load_sample, use_container_width=True)
    with b2:
        st.button("Clear", on_click=clear_input, use_container_width=True)
    with b3:
        analyze_clicked = st.button("Analyze content", type="primary", use_container_width=True)

with right:
    st.subheader("What you'll get")
    st.markdown("""
**Toxicity & hate speech**  
Detects abusive, hateful, or harassing language.

**Copyright risk**  
Estimates signals of possible unauthorized reproduction or similarity.

**Cultural sensitivity**  
Flags potentially insensitive stereotypes or disrespectful cultural references.
""")
    st.info("A copyright score cannot establish infringement on its own.")

if analyze_clicked:
    text = st.session_state.get("input_text", "").strip()
    if not text:
        st.warning("Enter some text before running the analysis.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to your local .env file or Streamlit Cloud Secrets.")
    else:
        try:
            from project import app as analysis_graph
            with st.status("Running the analysis branches...", expanded=True) as status:
                result = analysis_graph.invoke({"raw_input": text, "safety_scores": {}})
                status.update(label="Analysis complete", state="complete", expanded=False)

            scores = result.get("safety_scores", {})
            categories = [
                ("Toxicity & hate speech", "toxicity_level", "Abusive, hateful, or harassing language"),
                ("Copyright risk", "copyright_risk", "Potential unauthorized reproduction or similarity"),
                ("Cultural sensitivity", "cultural_insensitivity", "Potentially insensitive cultural references"),
            ]
            normalized = {}
            st.divider()
            st.subheader("Analysis results")
            columns = st.columns(3, gap="medium")
            for column, (label, key, note) in zip(columns, categories):
                try:
                    score = max(0, min(100, int(scores[key])))
                except (KeyError, TypeError, ValueError):
                    score = None
                normalized[key] = score
                with column:
                    shown = "N/A" if score is None else f"{score}/100"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-score">{shown}</div><div class="metric-note">{note}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if score is not None:
                        st.progress(score / 100, text=f"Estimated risk: {score}%")
                    else:
                        st.warning("No valid score returned by the model.")

            valid = [value for value in normalized.values() if value is not None]
            if valid:
                st.markdown("### Overall risk overview")
                st.metric("Average of the three scores", f"{round(sum(valid) / len(valid))}/100")
                st.caption("Simple average for display; not a validated risk metric.")

            with st.expander("View analyzed text"):
                st.write(text)

            report = (
                "CONTENT SAFETY ANALYSIS\n=======================\n\n"
                f"Toxicity & hate speech: {normalized['toxicity_level']}/100\n"
                f"Copyright risk: {normalized['copyright_risk']}/100\n"
                f"Cultural sensitivity: {normalized['cultural_insensitivity']}/100\n\n"
                f"Analyzed text:\n{text}\n"
            )
            st.download_button("Download results (.txt)", report, "content_safety_analysis.txt", "text/plain")
            st.caption("Scores are LLM-generated estimates. Review context before taking action.")
        except Exception as exc:
            st.error("The analysis could not be completed.")
            st.code(str(exc))
            st.info("Check project.py is beside app.py, its compiled graph is named `app`, dependencies are installed, and GROQ_API_KEY is configured.")
