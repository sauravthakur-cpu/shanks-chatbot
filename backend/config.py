"""
Central configuration for Shanks.
All tunable values live here so nothing is hard-coded deep inside the logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file placed at the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- LLM provider ---
# "ollama" (free, runs entirely on your Mac, no API key) / "anthropic" / "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama runs a local server on your Mac -- nothing leaves your machine, no cost, no key.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- Embeddings / retrieval ---
# Runs 100% locally and free -- no API key or internet needed after first download.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", 180))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 40))
TOP_K = int(os.getenv("TOP_K", 4))

# Similarity below this means "the knowledge base doesn't really cover this" ->
# Shanks falls back to a live web search instead of guessing.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))

# --- Paths ---
KNOWLEDGE_BASE_DIR = ROOT_DIR / "data" / "knowledge_base"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
FRONTEND_DIR = ROOT_DIR / "frontend"

# --- Web search fallback ---
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 4))

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
