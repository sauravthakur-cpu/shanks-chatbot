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
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

import config
from web_search import search_web

SYSTEM_PROMPT = """You are Shanks, a helpful, friendly assistant for college students, \
themed as a small, confident swordsman-adventurer mascot. You help students of Amizone \
(the college) with questions about their courses, portal, timetable, exams, fees, clubs, \
and general campus life, and you can also answer general questions using web results \
you're given.

Rules you always follow:
- Answer ONLY using the CONTEXT provided below. Do not use outside knowledge that \
contradicts it, and do not invent facts, numbers, deadlines, or policies that aren't in \
the context.
- If the context is genuinely insufficient to answer, say so clearly and suggest what the \
student should check or who to contact -- never guess.
- Be concise, direct, and warm. No filler, no fake enthusiasm, no invented testimonials \
or reviews.
- If you're using web results, make clear the information comes from the web, not the \
official college knowledge base, since it may be less precise for Amizone-specific details.
"""


class RAGEngine:
    def __init__(self):
        torch.set_num_threads(config.TORCH_THREADS)
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

        self.llm_client = self._init_llm_client()
        self._cache = OrderedDict()

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

    # ---------- generation ----------

    def _call_llm(self, user_question: str, context_block: str) -> str:
        user_message = f"CONTEXT:\n{context_block}\n\nSTUDENT QUESTION:\n{user_question}"

        if config.LLM_PROVIDER == "ollama":
            response = requests.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                                        "stream": False,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": {
                        "num_ctx": config.OLLAMA_NUM_CTX,
                        "num_predict": config.OLLAMA_NUM_PREDICT,
                        "temperature": config.OLLAMA_TEMPERATURE,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        elif config.LLM_PROVIDER == "anthropic":
            response = self.llm_client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=700,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        else:  # openai
            response = self.llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

    def answer(self, question: str) -> dict:
        key = " ".join(question.lower().split())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._answer_uncached(question)
        if result["source"] != "none":
            self._cache[key] = result
            if len(self._cache) > config.CACHE_SIZE:
                self._cache.popitem(last=False)
        return result

    def _answer_uncached(self, question: str) -> dict:
        kb_results = self.retrieve(question)
        best_score = kb_results[0]["score"] if kb_results else 0.0

        if best_score >= config.SIMILARITY_THRESHOLD:
            context_block = "\n\n---\n\n".join(
                f"(from {r['source']}):\n{r['text']}" for r in kb_results
            )
            reply = self._call_llm(question, context_block)
            return {
                "reply": reply,
                "source": "knowledge_base",
                "matches": [{"source": r["source"], "score": round(r["score"], 3)} for r in kb_results],
            }

        # Knowledge base wasn't confident -> fall back to a live web search
        web_results = search_web(question)
        if not web_results:
            return {
                "reply": (
                    "I couldn't find this in the college knowledge base, and a live web "
                    "search didn't return anything useful either. Could you rephrase the "
                    "question, or check with the college admin/faculty directly for this one?"
                ),
                "source": "none",
                "matches": [],
            }

        context_block = "\n\n---\n\n".join(
            f"(web result: {r['title']} -- {r['url']}):\n{r['snippet']}" for r in web_results
        )
        reply = self._call_llm(question, context_block)
        return {
            "reply": reply,
            "source": "web",
            "matches": [{"source": r["url"], "title": r["title"]} for r in web_results],
        }
