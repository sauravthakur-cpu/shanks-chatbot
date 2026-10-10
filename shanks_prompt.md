# Shanks Chatbot — Prompt File

> **Note:** The very first prompt you gave when the project was originally generated isn't visible to me, so I haven't copied it. This file is a **reconstructed prompt** written from your actual project files (code, config, README). If you paste it into an AI assistant, it should recreate a project that matches Shanks closely. It also includes the **runtime system prompt** that Shanks itself uses, copied word-for-word from `backend/rag_engine.py`.

---

## Part 1 — Master Build Prompt (reconstructed)

```text
Build me a complete, working RAG (Retrieval-Augmented Generation) chatbot called
"Shanks" for my college's student portal, Amizone (Amity University Gurugram).
It must run entirely on my Mac with no paid API keys.

GOAL
Students ask questions about attendance, exams, backlogs, fees, timetable and
portal usage. Shanks answers from a curated college knowledge base first. If the
knowledge base doesn't cover the question, it falls back to a live web search
instead of just saying "I don't know". It must never invent facts.

TECH STACK
- Backend: Python 3.11, FastAPI, Uvicorn
- Embeddings: sentence-transformers, model "all-MiniLM-L6-v2" (local, free)
- Vector store: FAISS (faiss-cpu), IndexFlatIP on L2-normalised vectors
  (inner product = cosine similarity)
- LLM: Ollama running llama3.1 locally (default). Also support Anthropic and
  OpenAI as optional providers, switched with LLM_PROVIDER in a .env file.
- Web fallback: the free, keyless "ddgs" DuckDuckGo library
- Frontend: plain HTML, CSS and vanilla JavaScript (no framework, no build step),
  served by FastAPI itself

PROJECT STRUCTURE
shanks-chatbot/
  backend/
    main.py           FastAPI app, POST /api/chat, serves the frontend
    rag_engine.py     retrieval + confidence check + fallback + LLM call
    ingest.py         builds the FAISS index from the knowledge base
    web_search.py     DuckDuckGo fallback
    config.py         every tunable value in one place, read from .env
    requirements.txt  pinned versions
  data/knowledge_base/amizone_faq.md   the college content (Markdown)
  frontend/ index.html, style.css, script.js, assets/
  vector_store/       generated index (git-ignored)
  .env.example, .gitignore, README.md

BEHAVIOUR
1. ingest.py reads every .md/.txt file in data/knowledge_base, splits it into
   overlapping word chunks (180 words, 40 overlap), embeds them once, and saves
   index.faiss + chunks.json into vector_store/.
2. On server startup, load the embedding model and the FAISS index ONCE.
3. For each question: embed it, retrieve TOP_K=4 chunks.
4. If the best similarity score >= SIMILARITY_THRESHOLD (0.35), answer from those
   chunks and label the reply "knowledge_base".
5. Otherwise, run a live web search (max 4 results), answer only from those
   results, and label the reply "web".
6. If the web search returns nothing, say so honestly and label it "none".
7. The API returns JSON: { reply, source, matches }.

CONFIG (config.py, all overridable via .env)
LLM_PROVIDER, OLLAMA_HOST, OLLAMA_MODEL, ANTHROPIC_MODEL, OPENAI_MODEL,
EMBEDDING_MODEL_NAME, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, TOP_K,
SIMILARITY_THRESHOLD, WEB_SEARCH_MAX_RESULTS, HOST, PORT.

FRONTEND
- Two-panel layout: a sidebar (mascot image, "Shanks" title, tagline
  "Your Amizone assistant", short blurb, legend with a "Knowledge base" badge and
  a "Web" badge) and a main chat panel (scrolling messages + input + Send button).
- User messages on the right in the accent colour, bot messages on the left.
- Every bot reply shows a coloured source badge above the text.
- Show "Shanks is thinking..." while waiting for the backend.
- On error, show a friendly message instead of crashing.
- Greet the user on load.

QUALITY RULES
- Keep ingestion, retrieval/generation, web search and API in separate modules.
- Never re-embed on every request.
- Web search must never crash a request: catch errors and return [].
- Add a README with setup steps for Mac (venv, pip install, Ollama pull,
  ingest, run uvicorn).
- Add a .gitignore that excludes .env, venv/, __pycache__/, and the generated
  vector store files.
- Pin all dependency versions in requirements.txt.

Give me every file in full, then step-by-step run instructions.
```

---

## Part 2 — Runtime System Prompt (copied from `backend/rag_engine.py`)

This is the prompt Shanks sends to the LLM with every question. The retrieved knowledge-base chunks or web results are added after it as the **CONTEXT**.

```text
You are Shanks, a helpful, friendly assistant for college students, themed as a
small, confident swordsman-adventurer mascot. You help students of Amizone (the
college) with questions about their courses, portal, timetable, exams, fees,
clubs, and general campus life, and you can also answer general questions using
web results you're given.

Rules you always follow:
- Answer ONLY using the CONTEXT provided below. Do not use outside knowledge that
  contradicts it, and do not invent facts, numbers, deadlines, or policies that
  aren't in the context.
- If the context is genuinely insufficient to answer, say so clearly and suggest
  what the student should check or who to contact -- never guess.
- Be concise, direct, and warm. No filler, no fake enthusiasm, no invented
  testimonials or reviews.
- If you're using web results, make clear the information comes from the web, not
  the official college knowledge base, since it may be less precise for
  Amizone-specific details.
```

---

## Part 3 — Follow-up Prompts Used While Building

These are the improvement requests made after the first version, rewritten as reusable prompts.

**1. Fill the knowledge base with real college content**
```text
Rewrite data/knowledge_base/amizone_faq.md for Amity University Gurugram.
Attendance: 75% required, 70% with B-category condonation. One exam attempt per
year; revaluation fee Rs 500. A backlog must be cleared the next year for
Rs 1000, whether the student failed or was debarred. Add fee payment, Amizone
portal FAQs, and contacts (Block A admission office for fees/timetable/penalty,
department for attendance/results, Block C for xerox and thesis printing).
Use clear Markdown headings, one topic per section.
```

**2. Make the web fallback more reliable**
```text
Casual questions like "whats a good way to get placed this year" sometimes return
zero DuckDuckGo results. Update web_search.py to: try the query as typed, retry
once after a short pause, then retry with a simplified keyword-only version
(strip punctuation and filler words). Never raise an exception. Log what was
tried when nothing is found.
```

**3. Re-theme the UI to match Amizone**
```text
Restyle the frontend to match the Amizone portal: navy blue (#1C3F94) sidebar,
gold (#E8A33D) accent, light (#F4F6FA) chat background, white input field,
white/light bot bubbles with dark text. Keep the green "Knowledge base" and red
"Web" badges. Use CSS variables in :root so the palette is easy to change.
```

**4. Swap and resize the mascot**
```text
Replace the SVG mascot with my own image (assets/shanks.jpg). Make the frame
wider, keep the image undistorted with object-fit: cover, and round the corners.
```

**5. Push to GitHub**
```text
Give me step-by-step commands to initialise git, commit, create a GitHub repo,
set the remote and push, making sure .env and venv/ are never committed.
```

---

## Part 4 — How to Run (quick reference)

```bash
# one-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
ollama pull llama3.1
python backend/ingest.py

# every time you want to use it
source venv/bin/activate
uvicorn main:app --reload --app-dir backend
# then open http://127.0.0.1:8000
```
