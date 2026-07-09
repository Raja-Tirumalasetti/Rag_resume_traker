# 🤖 Resume AI — System Prompts & Project Analysis

> **Project:** RAG-based Resume Analyzer  
> **Stack:** Python · Streamlit · ChromaDB · LangChain · OpenRouter  
> **Branch:** `ra`

---

## 📁 Project Architecture Overview

```
resume_analyzer/
├── app.py           ← Streamlit UI + main orchestration logic
├── capi.py          ← OpenRouter LLM API caller (LangChain wrapper)
├── chbd.py          ← ChromaDB vector store + hybrid search engine
├── chunk.py         ← PDF/DOCX text extraction + chunking pipeline
├── embeddeing.py    ← Local embedding model (all-MiniLM-L6-v2)
├── resumes/         ← Input folder: PDF & DOCX resume files
├── chroma_db/       ← Persisted vector database (auto-generated)
├── requirements.txt ← Python dependencies
└── .env             ← API keys (OPENROUTER_API_KEY)
```

---

## 🔄 Data Pipeline (End-to-End Flow)

```
resumes/ (PDF/DOCX)
        │
        ▼
   [chunk.py]  ──── extract_text() ────► raw text
        │
        ▼
   split_text_into_chunks()  (1 resume = 1 chunk, whole doc)
        │
        ▼
   process_resumes_folder()
   → { text, source, chunk_id, candidate }
        │
        ▼
   [chbd.py]  add_chunks_to_db()
        │
        ▼
   [embeddeing.py]  all-MiniLM-L6-v2  (local, no API key)
        │
        ▼
   ChromaDB  (persisted in chroma_db/)
        │
        ▼
   [hybrid_search()]  keyword + semantic merge
        │
        ▼
   Top-K context chunks  (K=24)
        │
        ▼
   [capi.py]  LLM prompt via OpenRouter
        │
        ▼
   Final Answer  ← displayed in Streamlit UI
```

---

## 🧠 System Prompts Used in the Application

### 1. 📋 Main Resume Q&A System Prompt
**Location:** `app.py` Lines 324–335  
**Triggered:** Every time user submits a question in the chat UI

```python
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
```

**Purpose:** Grounds the LLM strictly to retrieved resume content. Prevents hallucination.  
**Key Design Decisions:**
- Rule 2 forces a specific "not exist" sentinel → detected programmatically to show a user-friendly error
- No chain-of-thought or reasoning steps → direct, fast answers
- Context is injected as raw text blocks with `--- Resume: <source> ---` delimiters

---

### 2. 🔌 API Connection Test Prompt
**Location:** `app.py` Lines 198–208  
**Triggered:** When user clicks "🔌 Test API Connection" in sidebar

```python
test_resp = capi.query_openrouter(
    "Respond with only the word OK",
    model=chosen_model_id,
    api_key=st.session_state.or_key
)
```

**Purpose:** Minimal one-word prompt to validate LLM connectivity without wasting tokens.  
**Key Design Decisions:**
- Ultra-short prompt → cost efficient (minimal token usage)
- Checks if `"ok"` exists in response (case-insensitive) → tolerates minor formatting variations
- Tests the currently selected model — not a hardcoded model

---

## 🔍 Context Building Logic

**Location:** `app.py` Lines 312–322

Each retrieved document chunk is formatted as:
```
--- Resume: <filename>.pdf (Candidate: John Doe) ---
<full resume text content>

```

**Design Choices:**
- Candidate name and source file are prefixed → LLM can attribute answers to specific people
- `seen_sources` set tracks which files contributed → shown as source pills in UI
- Up to **K=24** chunks retrieved per query (set as `RETRIEVAL_K` constant)

---

## 🗄️ ChromaDB & Embedding Configuration

**Location:** `chbd.py` | `embeddeing.py`

| Setting | Value | Why |
|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (local) | No API key needed, fast, free |
| Vector Store | ChromaDB (persisted) | Local, no cloud needed |
| Collection Name | `resumes_collection` | Single collection for all resumes |
| Retrieval K | `24` | Wide net for hybrid search |
| Search Strategy | Hybrid (keyword + semantic) | Better recall for name/skill queries |

### Hybrid Search Scoring (keyword_search)

| Match Type | Score Boost |
|---|---|
| Keyword in page content | `+1.0 × frequency` |
| Keyword matches candidate name | `+100.0` |
| Keyword matches source filename | `+50.0` |

---

## 📦 Chunking Strategy

**Location:** `chunk.py`

```
CHUNK_SIZE    = 1500 chars   (defined but not used — whole resume = 1 chunk)
CHUNK_OVERLAP = 200  chars   (defined but not used currently)
```

> **Current Strategy:** Each resume file → **1 single chunk** (entire document).  
> `split_text_into_chunks()` returns the whole text as-is.

**Chunk Format (stored in ChromaDB):**
```
Candidate: <Name Extracted from Filename>
Source: <filename.pdf>

<Full Resume Text Content>
```

**Candidate Name Extraction (`clean_filename_to_name`):**
- Strips roll numbers (e.g., `21H51A6740`)
- Removes common keywords: `resume`, `updated`, `cv`, `btech`, etc.
- Splits on ` - ` to get candidate name portion
- Applies `.title()` formatting

---

## 🤖 Supported LLM Models (via OpenRouter)

**Location:** `app.py` Lines 16–27

| Display Name | Model ID |
|---|---|
| Gemini 2.5 Flash ⚡ | `google/gemini-2.5-flash` |
| Gemini 2.5 Pro 🧠 | `google/gemini-2.5-pro` |
| GPT-4o 🔥 | `openai/gpt-4o` |
| GPT-4o Mini ⚡ | `openai/gpt-4o-mini` |
| Claude 3.5 Sonnet 🎨 | `anthropic/claude-sonnet-4-5` |
| Claude 3.5 Haiku 🚀 | `anthropic/claude-haiku-4-5` |
| Llama 3.3 70B 🦙 | `meta-llama/llama-3.3-70b-instruct` |
| Llama 3 8B Free 🦙 | `meta-llama/llama-3-8b-instruct:free` |
| Mistral 7B Free 🌀 | `mistralai/mistral-7b-instruct:free` |
| DeepSeek V3 Free 🐳 | `deepseek/deepseek-chat-v3-0324:free` |

**LLM Call Settings:**
```python
temperature = 0.0      # Deterministic, factual answers
max_tokens  = 500      # Prevents 402 credit errors on OpenRouter
```

---

## 🛡️ "Not Found" Detection Logic

**Location:** `app.py` Lines 340–349

```python
is_missing = cleaned.lower().strip() in {
    "not exist", "not exist.", "not found", "not found.",
    "information not found", "information not available", "n/a",
}
```

If LLM returns one of these sentinel phrases (as instructed in Rule 2 of the system prompt), the UI shows:
> ❌ This information was not found in the indexed resumes.

And **sources are cleared** → clean UX with no false attribution.

---

## 🛠️ Dependencies

**Location:** `requirements.txt`

| Package | Purpose |
|---|---|
| `chromadb` | Vector database + local embeddings |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-dotenv` | `.env` file management |
| `openai` | OpenRouter compatibility (OpenAI API format) |
| `streamlit` | Web UI framework |
| `langchain` | LLM orchestration framework |
| `langchain-community` | Community integrations |
| `langchain-openai` | ChatOpenAI wrapper for OpenRouter |
| `langchain-chroma` | LangChain ↔ ChromaDB integration |

---

## 📝 Improvement Suggestions

### Prompting Improvements
- [ ] Add **multi-turn conversation** support (chat history context)
- [ ] Add **structured output** (JSON mode) for skill extraction queries
- [ ] Add a **ranking/comparison** prompt for "who is best for X role?" queries
- [ ] Increase `max_tokens` to `1000–2000` for longer, richer answers

### RAG Pipeline Improvements
- [ ] Enable **actual chunking** (CHUNK_SIZE=1500, CHUNK_OVERLAP=200) for large resumes
- [ ] Add **metadata filters** (filter by candidate name before similarity search)
- [ ] Add **re-ranking** step after hybrid search (cross-encoder model)
- [ ] Support **multi-file** resume uploads via the UI

### Architecture Improvements
- [ ] Fix typo: `embeddeing.py` → `embedding.py`
- [ ] Add **async processing** for large resume batches
- [ ] Add **cache layer** for repeated identical queries
- [ ] Add **logging** for audit trail of queries and answers

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API Key in .env
echo OPENROUTER_API_KEY=sk-or-v1-... > .env

# 3. Add resumes to resumes/ folder (PDF or DOCX)

# 4. Run the app
streamlit run app.py

# 5. Click "Index / Re-index Resumes" in sidebar
# 6. Ask questions in the chat interface!
```

---

*Generated by analyzing: `app.py`, `capi.py`, `chbd.py`, `chunk.py`, `embeddeing.py`, `requirements.txt`, `.env`*
