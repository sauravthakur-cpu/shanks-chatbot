"""
Entry point. Run with:

    uvicorn backend.main:app --reload

from the project root (with your virtual environment active).
"""

import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag_engine import RAGEngine

app = FastAPI(title="Shanks - College RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local/dev use; tighten this before any public deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request -- this is what keeps response latency low.
engine: RAGEngine | None = None


@app.on_event("startup")
def load_engine():
    global engine
    engine = RAGEngine()
    # Warm up in the background so the server starts at once but the first question is still fast.
    threading.Thread(target=engine.warm_up, daemon=True).start()
    print("Shanks is ready.")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    source: str
    matches: list


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine still starting up, try again shortly")
    result = engine.answer(req.message.strip())
    return result


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine still starting up, try again shortly")

    def event_lines():
        # One JSON object per line (NDJSON) -- the browser reads them as they arrive.
        for event in engine.answer_stream(message):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_lines(), media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_ready": engine is not None}


# Serve the frontend (index.html, style.css, script.js, assets/) directly,
# so opening http://127.0.0.1:8000 gives you the whole app -- no separate server needed.
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
#comment 