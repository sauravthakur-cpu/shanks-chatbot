"""
The brain of Shanks.

Flow for every incoming message:
  1. Embed the question, search the local FAISS index (the college knowledge base).
  2. If the best match is confident enough -> answer grounded in that content.
  3. If not -> fall back to a live web search, and answer grounded in that instead.
  4. Either way, the LLM is only allowed to answer from the material it was
     actually given -- it is told to say so plainly if neither source helps,
     instead of inventing an answer.

This keeps answers honest and traceable: every reply says whether it came
from the knowledge base or the web, so nothing is presented as certain when
it isn't.
"""

import json
import os
import re
import time
from collections import OrderedDict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

import config
from web_search import search_web

# Stop PyTorch grabbing every CPU core -- leaves room for Ollama and the rest of the laptop.
torch.set_num_threads(config.TORCH_THREADS)

SYSTEM_PROMPT = """You are Shanks, a cheerful swordsman-adventurer mascot who helps students of \
Amizone (Amity University). Be warm and upbeat (at most one emoji) but precise. Answer ONLY from \
the CONTEXT given. Never invent facts, numbers, deadlines or policies. If the context is not \
enough, say so and suggest who to contact. Be concise: at most 4 short sentences or 5 bullet \
points. If the context comes from the web, say so, since it may be less precise for \
Amizone-specific details."""

_HELLO = re.compile(r"^(hi|hello|hey|yo|hola|namaste|good (morning|afternoon|evening))\W*$", re.I)
_THANKS = re.compile(r"^(thanks|thank you|thx|ty)\b", re.I)


def small_talk(text: str):
    """Instant replies for greetings/thanks -- no search, no LLM, no waiting."""
    t = text.strip()
    if _HELLO.match(t):
        return ("Hey there, I'm Shanks! \u2694\ufe0f Ask me about attendance, exams, backlog or fees, "
                "hall tickets, contacts or hostel.")
    if _THANKS.match(t):
        return "Anytime! Shout if you need anything else. \u2694\ufe0f"
    return None


class RAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        index_path = config.VECTOR_STORE_DIR / "index.faiss"
        chunks_path = config.VECTOR_STORE_DIR / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "No vector index found. Run `python backend/ingest.py` first "
                "(after adding your files to data/knowledge_base/)."
            )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.session = requests.Session()  # reuses the connection to Ollama between questions
        self.cache = OrderedDict()         # question -> answer, for instant repeat answers
        self.llm_client = self._init_llm_client()

    def _init_llm_client(self):
        if config.LLM_PROVIDER == "ollama":
            # Nothing to initialize -- just confirm the local Ollama server is reachable,
            # so we fail fast with a clear message instead of erroring on the first question.
            try:
                requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    "LLM_PROVIDER=ollama but no Ollama server is running. "
                    "Install it from https://ollama.com/download, then run "
                    f"`ollama pull {config.OLLAMA_MODEL}` and try again."
                )
            return None
        elif config.LLM_PROVIDER == "anthropic":
            import anthropic
            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env")
            return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        elif config.LLM_PROVIDER == "openai":
            import openai
            if not config.OPENAI_API_KEY:
                raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env")
            return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")

    # ---------- retrieval ----------

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or config.TOP_K
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk["text"], "source": chunk["source"], "score": float(score)})
        return results

    # ---------- warm-up ----------

    def warm_up(self):
        """Load the embedder and the LLM into memory now, so the FIRST student question is fast."""
        try:
            self.embedder.encode(["warm up"], normalize_embeddings=True)
            if config.LLM_PROVIDER == "ollama":
                self.session.post(
                    f"{config.OLLAMA_HOST}/api/chat",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": config.OLLAMA_KEEP_ALIVE,
                        # must match the real requests, or Ollama reloads the model
                        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 1},
                    },
                    timeout=180,
                )
            print("[warm_up] models are loaded and ready.")
        except Exception as exc:
            print(f"[warm_up] skipped: {exc}")

    # ---------- generation (streaming) ----------

    def _stream_llm(self, user_question: str, context_block: str):
        """Yields the answer piece by piece, as the model produces it."""
        user_message = f"CONTEXT:\n{context_block}\n\nSTUDENT QUESTION:\n{user_question}"

        if config.LLM_PROVIDER == "ollama":
            payload = {
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": True,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {
                    "num_ctx": config.OLLAMA_NUM_CTX,
                    "num_predict": config.OLLAMA_NUM_PREDICT,
                    "temperature": config.OLLAMA_TEMPERATURE,
                },
            }
            with self.session.post(
                f"{config.OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=(5, 120)
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break

        elif config.LLM_PROVIDER == "anthropic":
            with self.llm_client.messages.stream(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.OLLAMA_NUM_PREDICT,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for piece in stream.text_stream:
                    yield piece

        else:  # openai
            stream = self.llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=config.OLLAMA_NUM_PREDICT,
                stream=True,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    # ---------- main entry points ----------

    def answer_stream(self, question: str):
        """
        Generator of events:
          {"type": "meta", "source": ..., "matches": [...]}   (once, first)
          {"type": "token", "text": "..."}                    (many)
          {"type": "done"}                                    (once, last)
        """
        t0 = time.perf_counter()
        key = " ".join(question.lower().split())

        # Repeat question? Answer instantly from memory.
        if key in self.cache:
            self.cache.move_to_end(key)
            cached = self.cache[key]
            print(f"[timing] cache hit ({time.perf_counter() - t0:.3f}s)")
            yield {"type": "meta", "source": cached["source"], "matches": cached["matches"]}
            yield {"type": "token", "text": cached["reply"]}
            yield {"type": "done"}
            return

        talk = small_talk(question)
        if talk:
            yield {"type": "meta", "source": "", "matches": []}
            yield {"type": "token", "text": talk}
            yield {"type": "done"}
            return

        kb_results = self.retrieve(question)
        best_score = kb_results[0]["score"] if kb_results else 0.0
        print(f"[timing] retrieval {time.perf_counter() - t0:.3f}s (best score {best_score:.2f})")

        if best_score >= config.SIMILARITY_THRESHOLD:
            source = "knowledge_base"
            context_block = "\n\n---\n\n".join(
                f"(from {r['source']}):\n{r['text']}" for r in kb_results
            )
            matches = [{"source": r["source"], "score": round(r["score"], 3)} for r in kb_results]
        else:
            t1 = time.perf_counter()
            web_results = search_web(question)
            print(f"[timing] web search {time.perf_counter() - t1:.2f}s ({len(web_results)} results)")
            if not web_results:
                yield {"type": "meta", "source": "none", "matches": []}
                yield {"type": "token", "text": (
                    "I couldn't find this in the college knowledge base, and a live web "
                    "search didn't return anything useful either. Could you rephrase the "
                    "question, or check with the college admin/faculty directly for this one?"
                )}
                yield {"type": "done"}
                return
            source = "web"
            context_block = "\n\n---\n\n".join(
                f"(web result: {r['title']} -- {r['url']}):\n{r['snippet']}" for r in web_results
            )
            matches = [{"source": r["url"], "title": r["title"]} for r in web_results]

        yield {"type": "meta", "source": source, "matches": matches}

        parts = []
        first_token_logged = False
        try:
            for piece in self._stream_llm(question, context_block):
                if not first_token_logged:
                    print(f"[timing] first token at {time.perf_counter() - t0:.2f}s")
                    first_token_logged = True
                parts.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:
            print(f"[llm] error: {exc}")
            yield {"type": "token", "text": "\n\n(Sorry, the language model stopped responding. "
                                            "Please make sure Ollama is running and try again.)"}
            yield {"type": "done"}
            return

        print(f"[timing] total {time.perf_counter() - t0:.2f}s")

        # Remember knowledge-base answers only (web answers can go stale).
        if source == "knowledge_base" and parts:
            self.cache[key] = {"reply": "".join(parts), "source": source, "matches": matches}
            while len(self.cache) > config.CACHE_SIZE:
                self.cache.popitem(last=False)

        yield {"type": "done"}

    def answer(self, question: str) -> dict:
        """Non-streaming version (kept so /api/chat still works): collects the stream."""
        meta, parts = {}, []
        for event in self.answer_stream(question):
            if event["type"] == "meta":
                meta = event
            elif event["type"] == "token":
                parts.append(event["text"])
        return {
            "reply": "".join(parts),
            "source": meta.get("source", "none"),
            "matches": meta.get("matches", []),
        }
#comment