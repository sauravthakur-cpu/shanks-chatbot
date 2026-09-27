"""
Run this whenever you add or edit files in data/knowledge_base/.

    python backend/ingest.py

It reads every .md / .txt file in the knowledge base folder, splits it into
overlapping chunks, embeds each chunk locally with sentence-transformers,
and saves a FAISS index + the chunk text/metadata to vector_store/.

This is a separate, one-time (or "whenever content changes") step, kept apart
from the live chat server, so the server itself never pays the cost of
re-reading and re-embedding files on every request.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_documents(folder: Path) -> list[dict]:
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping some overlap for context continuity
    return chunks


def build_index():
    docs = load_documents(config.KNOWLEDGE_BASE_DIR)
    if not docs:
        print(f"No .md/.txt files found in {config.KNOWLEDGE_BASE_DIR}")
        print("Add your real Amizone / college content there first, then re-run this script.")
        sys.exit(1)

    print(f"Loaded {len(docs)} document(s). Chunking...")
    records = []  # one entry per chunk: {text, source}
    for doc in docs:
        for chunk in chunk_text(doc["text"], config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS):
            records.append({"text": chunk, "source": doc["source"]})

    print(f"Created {len(records)} chunks. Loading embedding model "
          f"'{config.EMBEDDING_MODEL_NAME}' (first run downloads it, ~90MB)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.VECTOR_STORE_DIR / "index.faiss"))
    with open(config.VECTOR_STORE_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. Index saved to {config.VECTOR_STORE_DIR}")
    print(f"{len(records)} chunks indexed from {len(docs)} source file(s).")


if __name__ == "__main__":
    build_index()
