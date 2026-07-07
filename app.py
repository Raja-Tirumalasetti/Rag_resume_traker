import os
import streamlit as st
from dotenv import load_dotenv, set_key

# Load .env with explicit absolute path
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)

import capi
import chunk
import chbd

# ── Constants ──────────────────────────────────────────────────────────────────
RETRIEVAL_K = 24

MODELS = {
    "Gemini 2.5 Flash ⚡":    "google/gemini-2.5-flash",
    "Gemini 2.5 Pro 🧠":      "google/gemini-2.5-pro",
    "GPT-4o 🔥":              "openai/gpt-4o",
    "GPT-4o Mini ⚡":         "openai/gpt-4o-mini",
    "Claude 3.5 Sonnet 🎨":   "anthropic/claude-sonnet-4-5",
    "Claude 3.5 Haiku 🚀":    "anthropic/claude-haiku-4-5",
    "Llama 3.3 70B 🦙":       "meta-llama/llama-3.3-70b-instruct",
    "Llama 3 8B Free 🦙":     "meta-llama/llama-3-8b-instruct:free",
    "Mistral 7B Free 🌀":     "mistralai/mistral-7b-instruct:free",
    "DeepSeek V3 Free 🐳":    "deepseek/deepseek-chat-v3-0324:free",
}
MODEL_NAMES  = list(MODELS.keys())
MODEL_IDS    = list(MODELS.values())

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp { background: linear-gradient(135deg, #0f0c29 0%, #141428 50%, #0f0c29 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13122b 0%, #1a1840 100%) !important;
    border-right: 1px solid rgba(77,150,255,0.15);
}
[data-testid="stSidebar"] * { color: #c8cde8 !important; }

.sec-title {
    font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #4D96FF !important; margin: 1rem 0 0.3rem 0;
    border-bottom: 1px solid rgba(77,150,255,0.2); padding-bottom: 3px;
}
.badge { padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-ok  { background: rgba(46,204,113,0.2); color: #2ecc71 !important; }
.badge-err { background: rgba(231,76,60,0.2);  color: #e74c3c !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4D96FF, #a855f7) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}

/* Response box */
.response-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(77,150,255,0.2);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    line-height: 1.7;
    color: #e8eaed;
}
.response-box h3 {
    color: #4D96FF; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 0.8rem;
}
.source-pill {
    background: rgba(255,142,83,0.12); border: 1px solid rgba(255,142,83,0.25);
    border-radius: 20px; padding: 3px 10px; font-size: 0.73rem; color: #fdba74;
    margin-right: 4px;
}
.not-found { color: #f87171; font-style: italic; }

/* Text input styling */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(77,150,255,0.3) !important;
    border-radius: 12px !important;
    color: #e8eaed !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #4D96FF !important;
    box-shadow: 0 0 0 3px rgba(77,150,255,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "or_key" not in st.session_state:
    st.session_state.or_key = os.getenv("OPENROUTER_API_KEY", "")
if "sel_model_name" not in st.session_state:
    st.session_state.sel_model_name = MODEL_NAMES[0]
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "db_count" not in st.session_state:
    st.session_state.db_count = 0

# ── DB Count ───────────────────────────────────────────────────────────────────
def refresh_db_count():
    try:
        st.session_state.db_count = len(chbd.get_chroma_db().get().get("documents", []))
    except Exception:
        st.session_state.db_count = 0

refresh_db_count()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Resume AI")

    # API Key
    st.markdown('<div class="sec-title">🔑 API Key</div>', unsafe_allow_html=True)
    key_input = st.text_input(
        "API Key", value=st.session_state.or_key,
        type="password", label_visibility="collapsed",
        placeholder="sk-or-v1-...",
    )
    if key_input != st.session_state.or_key:
        st.session_state.or_key = key_input
        if key_input.strip():
            set_key(_ENV_PATH, "OPENROUTER_API_KEY", key_input)

    badge = ('<span class="badge badge-ok">✓ Set</span>'
             if st.session_state.or_key
             else '<span class="badge badge-err">✗ Missing</span>')
    st.markdown(f"Status: {badge}", unsafe_allow_html=True)

    # Knowledge base
    st.markdown('<div class="sec-title">🗄️ Knowledge Base</div>', unsafe_allow_html=True)
    resumes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumes")
    resumes_list = (
        [f for f in os.listdir(resumes_dir) if f.lower().endswith((".pdf", ".docx"))]
        if os.path.exists(resumes_dir) else []
    )
    st.markdown(f"📁 **{len(resumes_list)}** resume files")
    st.markdown(f"📊 **{st.session_state.db_count}** chunks indexed")

    # Index button
    st.markdown('<div class="sec-title">⚡ Actions</div>', unsafe_allow_html=True)
    if st.button("🔄 Index / Re-index Resumes", use_container_width=True):
        if not resumes_list:
            st.error("No PDF/DOCX files in resumes/ folder.")
        else:
            prog = st.progress(0, text="Starting...")
            try:
                prog.progress(20, text="Extracting text...")
                chunks_list = chunk.process_resumes_folder(resumes_dir)
                prog.progress(55, text=f"Embedding {len(chunks_list)} chunks...")
                chbd.delete_collection()
                chbd.add_chunks_to_db(chunks_list)
                prog.progress(100, text="Done!")
                st.success(f"Indexed {len(chunks_list)} chunks from {len(resumes_list)} resumes!")
                refresh_db_count()
                st.rerun()
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()
    if st.button("🗑️ Clear Last Response", use_container_width=True):
        st.session_state.last_question = ""
        st.session_state.last_answer   = ""
        st.session_state.last_sources  = []
        st.rerun()


# ── MAIN AREA ──────────────────────────────────────────────────────────────────

# Header row: title left, model selector right
col_title, col_model = st.columns([3, 2])

with col_title:
    st.markdown("""
<div style="padding: 0.4rem 0 1rem 0;">
    <h1 style="background: linear-gradient(90deg,#FF6B6B,#FF8E53,#4D96FF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               font-size:2rem;font-weight:700;margin:0;">🤖 Resume AI</h1>
    <p style="color:#6b7280;font-size:0.9rem;margin:2px 0 0 0;">
        Ask anything about candidates in the database
    </p>
</div>
""", unsafe_allow_html=True)

with col_model:
    st.markdown('<p style="color:#4D96FF;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">🤖 Select Model</p>', unsafe_allow_html=True)
    chosen_name = st.selectbox(
        "Model", options=MODEL_NAMES,
        index=MODEL_NAMES.index(st.session_state.sel_model_name),
        key="model_dropdown",
        label_visibility="collapsed",
    )
    if chosen_name != st.session_state.sel_model_name:
        st.session_state.sel_model_name = chosen_name
        st.rerun()

st.divider()

# Warnings
if not st.session_state.or_key:
    st.warning("⚠️ Enter your OpenRouter API key in the sidebar.")
    st.stop()

if st.session_state.db_count == 0:
    st.info("📋 Click **Index / Re-index Resumes** in the sidebar to load the knowledge base first.")
    st.stop()

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a Question")

with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input(
        "Question",
        placeholder="e.g. What are Indira's skills?  /  Who knows Python & Django?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍 Search", use_container_width=False)

# ── Process Query ──────────────────────────────────────────────────────────────
if submitted and user_query.strip():
    model_id = MODELS[st.session_state.sel_model_name]

    with st.spinner("🔍 Searching resumes..."):
        try:
            db             = chbd.get_chroma_db()
            retriever      = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            retrieved_docs = retriever.invoke(user_query)

            if not retrieved_docs:
                st.session_state.last_question = user_query
                st.session_state.last_answer   = "❌ No matching resume content found."
                st.session_state.last_sources  = []
            else:
                context      = ""
                seen_sources: set[str] = set()
                for doc in retrieved_docs:
                    src       = doc.metadata.get("source", "Unknown")
                    candidate = doc.metadata.get("candidate", "")
                    context  += (
                        f"--- Resume: {src}"
                        + (f" (Candidate: {candidate})" if candidate else "")
                        + f" ---\n{doc.page_content}\n\n"
                    )
                    seen_sources.add(src)

                prompt = (
                    "You are a resume-intelligence assistant.\n"
                    "Answer questions about candidates using ONLY the resume content below.\n\n"
                    "RULES:\n"
                    "1. Use ONLY information from the resume chunks. Never hallucinate.\n"
                    "2. If information is not present in ANY chunk, respond ONLY with: not exist\n"
                    "3. For specific person questions, focus on their resume chunks.\n"
                    "4. Use bullet points for skills, projects, or experiences.\n"
                    "5. Be accurate, complete, and concise.\n\n"
                    f"Resume Chunks:\n{context}\n"
                    f"Question: {user_query}\n\nAnswer:"
                )

                raw = capi.query_openrouter(prompt, model=model_id, api_key=st.session_state.or_key)
                cleaned = raw.strip()

                is_missing = cleaned.lower().strip() in {
                    "not exist", "not exist.", "not found", "not found.",
                    "information not found", "information not available", "n/a",
                }

                st.session_state.last_question = user_query
                st.session_state.last_sources  = sorted(seen_sources)
                if is_missing:
                    st.session_state.last_answer = "❌ This information was not found in the indexed resumes."
                    st.session_state.last_sources = []
                else:
                    st.session_state.last_answer = cleaned

        except Exception as e:
            st.session_state.last_question = user_query
            st.session_state.last_answer   = f"⚠️ Error: {e}"
            st.session_state.last_sources  = []

# ── Show Response ──────────────────────────────────────────────────────────────
if st.session_state.last_answer:
    st.markdown(f"**Q: {st.session_state.last_question}**")
    st.markdown(
        f'<div class="response-box"><h3>Answer</h3>{st.session_state.last_answer}</div>',
        unsafe_allow_html=True
    )
    if st.session_state.last_sources:
        pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in st.session_state.last_sources)
        st.markdown(f'<div style="margin-top:10px">{pills}</div>', unsafe_allow_html=True)
