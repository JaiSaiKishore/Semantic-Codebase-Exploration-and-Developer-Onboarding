# 🧠 Semantic Codebase RAG: Developer Onboarding Assistant

> **An autonomous, syntax-aware codebase comprehension engine** combining multi-language **Concrete Syntax Tree (CST/AST)** parsing, **dual-collection dense vector retrieval**, and a **3-stage adaptive LangGraph reasoning pipeline**.

---

## 🌟 Key Features

1. **Multi-Language Tree-sitter AST Extraction:**
   - Native Concrete Syntax Tree parsing for **Python (`.py`)**, **JavaScript (`.js`, `.jsx`)**, **TypeScript (`.ts`, `.tsx`)**, and **Markdown (`.md`)**.
   - Extracts complete, syntactically intact classes, functions, and methods with qualified scopes and line bounds—zero syntax truncation.
   - Adaptive windowed fallback parser for C, C++, Java, Go, Rust, HTML, and CSS.

2. **Dual-Collection ChromaDB Architecture:**
   - **`codebase_chunks`**: Stores AST functional blocks and architectural documentation.
   - **`github_history`**: Stores closed GitHub Issues, Pull Requests, and bug discussions mined via PyGithub.

3. **Local ONNX Vector Embeddings (`FastEmbed`):**
   - High-throughput 384-dimensional dense vectors generated locally using `BAAI/bge-small-en-v1.5`.
   - 100% private, zero remote embedding API costs, zero rate limits.

4. **3-Stage Adaptive LangGraph Reasoning Pipeline:**
   - **Stage 1 (Direct Retrieval & Cross-Encoder Re-Ranking):** Evaluates vector matches with `ms-marco-MiniLM-L-6-v2` applying calibrated logit thresholds.
   - **Stage 2 (LLM Keyword Expansion):** Rescues vocabulary-mismatched or zero-issue queries by extracting 5–8 targeted technical keywords from codebase metadata and re-sweeping both collections.
   - **Stage 3 (Synthesis vs. Actionable Fallback):** Synthesizes grounded answers referencing exact code lines, or outputs a transparent retrieval trace if the query is unrelated.

5. **Real-Time Telemetry & Progress UI:**
   - Animated 4-step progress bar: `1. Clone` → `2. AST Parse` → `3. FastEmbed` → `4. Issues/PRs`.
   - Real-time 800ms telemetry polling, reasoning trace accordion, and collapsible retrieved code drawer.
   - Persistent multi-turn conversation memory via asynchronous SQLite checkpointer (`checkpoints.db`).
   - Clean session isolation on **+ New Chat** (`resetRepoBoxUI`).

6. **Enterprise CI/CD & Zero-Leak Administrative Tools:**
   - `POST /webhook/github`: Incremental git-pull and AST delta updates verified with HMAC SHA-256.
   - `python reset_db.py`: Complete system wipe (clears ChromaDB, deletes cloned repos with Windows `.git` permission handling, and resets chat history).
   - `python delete_repo.py`: Interactive CLI to delete a single repository.

---

## 📋 System Prerequisites

Before setting up the project, ensure you have:
- **Operating System:** Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+).
- **Python:** Version `3.10`, `3.11`, or `3.12` installed ([Download Python](https://www.python.org/downloads/)).
- **Git CLI:** Installed and accessible in your system PATH ([Download Git](https://git-scm.com/)).
- **Groq API Key (Free):** Required for the LangGraph agent. Get one instantly at [console.groq.com/keys](https://console.groq.com/keys).

---

## 🚀 How to Replicate and Run on Another Computer

Follow these 4 simple steps to set up and run this project on any machine:

### Step 1: Clone the Repository
```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd "codebase RAG"
```

### Step 2: Create and Activate a Python Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (`.env`)
Copy `.env.example` to `.env`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your free Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```
*(Optional)* Add a `GITHUB_TOKEN` to increase your GitHub API rate limit for large repositories.

---

## 💻 Running the Application

### Option A: One-Click Startup Scripts
- **Windows:** Double-click `run.bat` or run:
  ```powershell
  .\run.bat
  ```
  *(or PowerShell script: `.\run.ps1`)*
- **Linux / macOS:** Run:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### Option B: Manual Terminal Execution
With your virtual environment activated:
```bash
python server.py
```
Then open your browser and navigate to:
```
http://localhost:8000
```

---

## 🎮 How to Use the Application

1. **Index a Codebase:**
   - In the **Set Repo** input box on the web interface, paste any public GitHub repository URL (e.g. `https://github.com/patrickloeber/flask-todo` or `https://github.com/sindresorhus/yocto-queue`).
   - Click **Set Repo**. Observe the real-time progress bar move through:
     `1. Clone` → `2. AST Parse` → `3. FastEmbed` → `4. Issues/PRs` until reaching `100%`.
2. **Explore and Chat:**
   - Click one of the **Sample Exploration Prompts** (e.g., *Codebase Overview*, *Core Classes & Logic*, *Execution Flow*, or *Issues & PR History*), or type your own question.
   - The assistant responds with line-referenced code explanations.
3. **Inspect Behind the Scenes:**
   - Click **▼ Routing & Reasoning Trace** to see the pipeline's execution path.
   - Click **▼ 🔍 View Retrieved & Re-ranked Context** to see the raw AST blocks and line numbers retrieved from ChromaDB.
4. **Start a Fresh Chat:**
   - Click **+ New Chat** in the left sidebar to clear the conversation and reset the input box for a new repository.

---

## 🛠️ Administrative & Utility Scripts

- **Full System Wipe:**
  Wipes all ChromaDB vectors, removes cloned repositories, and resets conversation history:
  ```bash
  python reset_db.py
  ```
- **Delete Specific Repository:**
  Interactively select or specify a repository to delete:
  ```bash
  python delete_repo.py
  # or directly:
  python delete_repo.py owner/repo
  ```
- **Test Webhook CI/CD Sync:**
  Simulates a GitHub push event with HMAC SHA-256 verification:
  ```bash
  python test_webhook.py
  ```
- **Direct CLI Vector Search:**
  Run direct similarity queries against ChromaDB from the terminal:
  ```bash
  python test_search.py "queue implementation" yocto-queue
  ```

---

## 📂 Project Architecture & Directory Structure

```
codebase RAG/
├── agent.py               # LangGraph 3-stage adaptive reasoning pipeline & LLM synthesis
├── api.py                 # Secondary ASGI entrypoint re-exporting the FastAPI app
├── chunk_code.py          # Multi-language Tree-sitter AST parser (Py, JS, TS, MD) + window fallback
├── db_config.py           # Unified ChromaDB persistent client & LocalFastEmbed singleton
├── delete_repo.py         # Utility to delete a single repository from vector DB and disk
├── ingest.py              # Shallow git clone controller & PyGithub issue/PR miner
├── reset_db.py            # Complete system wipe utility (handles Windows .git locks)
├── requirements.txt       # Production dependency manifest
├── run.bat                # Windows 1-click batch launcher
├── run.ps1                # Windows PowerShell launcher
├── run.sh                 # Linux / macOS launcher
├── server.py              # Uvicorn FastAPI startup script (port 8000)
├── store_vectors.py       # Batch vector indexing pipeline
├── test_search.py         # Terminal CLI vector search utility
├── test_webhook.py        # GitHub webhook push event simulation with HMAC SHA-256
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore configuration (excludes runtime DBs and repos)
├── PROJECT_REPORT_COMPLETE.txt # Complete academic project report & viva question bank
├── backend/
│   ├── __init__.py
│   ├── ingestion.py       # Real-time 4-step progress telemetry service (INDEX_STATUS)
│   └── routes.py          # FastAPI REST endpoints, Webhook handler, static mounting
└── frontend/
    └── index.html         # Modern SPA interface (Tailwind-style dark mode, progress bar, chat)
```

---

## ❓ Troubleshooting & FAQs

- **Q: `ModuleNotFoundError: No module named 'tree_sitter_python'`**
  - **A:** Ensure you activated the virtual environment and ran `pip install -r requirements.txt`. The tree-sitter language bundles are pre-compiled and install automatically.
- **Q: `PermissionError: [WinError 5] Access is denied` on `./repos`**
  - **A:** Windows marks Git packfiles as read-only. Both `reset_db.py` and `delete_repo.py` include an automated `remove_readonly` error handler using `os.chmod(path, stat.S_IWRITE)`. Always use these scripts to delete repositories.
- **Q: Rate limit warning during GitHub issue fetching**
  - **A:** If you do not provide a `GITHUB_TOKEN`, GitHub limits unauthenticated requests to 60/hr. The system automatically falls back to unauthenticated mode without crashing. To increase limits, create a free personal token at [github.com/settings/tokens](https://github.com/settings/tokens) and paste it into `.env`.

