"""
graph/config.py
───────────────
Single source of truth for all configuration: env vars, model names,
retriever settings, and tunable thresholds.

Every other module in the graph package imports from here.
Nothing reads os.environ directly except this file.

Usage
─────
    from graph.config import settings

    llm = ChatGroq(model=settings.LLM_MODEL, ...)
    engine = create_async_engine(settings.BOT_DB_URL)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env once, at import time ───────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


# ── Validate required keys — fail loud, not silently ─────────────────────────
_REQUIRED = ["GROQ_API_KEY", "BOT_DB_URL"]
_missing = [k for k in _REQUIRED if not os.getenv(k)]
if _missing:
    raise EnvironmentError(
        f"❌ Missing required environment variables: {', '.join(_missing)}\n"
        "Check your .env file or Render dashboard."
    )


# ── LangSmith (optional — degrades gracefully if keys absent) ────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ClickPost-Control-Tower-Beta")


@dataclass(frozen=True)
class Settings:
    """Immutable configuration loaded once at startup."""

    # ── Secrets & connection strings ─────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: os.environ["GROQ_API_KEY"])

    BOT_DB_URL: str = field(default_factory=lambda: os.environ["BOT_DB_URL"])
    """PostgreSQL connection string for PGVector.
    Renamed from DATABASE_URL to prevent Chainlit from hijacking it
    as its own data layer (causes 'relation User does not exist' errors).
    Format: postgresql+psycopg://user:pass@host:5432/db?sslmode=require
    """

    SUPABASE_FEEDBACK_URL: str = field(
        default_factory=lambda: os.getenv("SUPABASE_FEEDBACK_URL", "")
    )
    """Separate connection string for the ct_feedback table.
    Falls back to empty string — feedback writes silently skip if unset.
    """

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_RETRIES: int = 3

    # ── Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    """Defaults to MiniLM (fits Render 512MB).
    Override in .env with EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 for local dev
    if your Supabase collection was ingested with BAAI embeddings.
    Must match the model used at ingest time — mismatched embeddings = garbage retrieval.
    """

    COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv(
            "COLLECTION_NAME", "control_tower_docs_minilm"
        )
    )
    """PGVector collection name. Must match what ingest_render.py created.
    Defaults to MiniLM collection. Override for BAAI: control_tower_docs
    """

    # ── Retriever ────────────────────────────────────────────────────────
    SEARCH_TYPE: str = "mmr"
    RETRIEVER_K: int = 6
    """Number of documents returned to the grader."""

    RETRIEVER_FETCH_K: int = 30
    """Candidate pool size for MMR diversity reranking."""

    RETRIEVER_LAMBDA: float = 0.6
    """MMR lambda — 1.0 = pure relevance, 0.0 = pure diversity.
    0.6 balances well for the Control Tower knowledge base.
    """

    # ── Graph behavior ───────────────────────────────────────────────────
    MAX_REWRITE_RETRIES: int = 2
    """Max retrieve→grade→rewrite cycles before generator runs with
    whatever filtered_docs it has (possibly empty).
    """

    MEMORY_WINDOW: int = 6
    """Number of recent (human, ai) message pairs kept in chat_history.
    6 pairs = 12 messages. Enough for multi-turn context without
    blowing up the Groq context window (128k tokens, but we want
    to leave room for retrieved docs + system prompt).
    """

    MIN_RELEVANT_DOCS: int = 1
    """Minimum filtered_docs needed to skip the rewrite loop.
    If grading leaves fewer than this, the rewriter fires.
    """

    @property
    def retriever_kwargs(self) -> dict:
        """Pre-built kwargs dict for PGVector.as_retriever()."""
        return {
            "k": self.RETRIEVER_K,
            "fetch_k": self.RETRIEVER_FETCH_K,
            "lambda_mult": self.RETRIEVER_LAMBDA,
        }


# ── Module-level singleton — import this everywhere ──────────────────────────
settings = Settings()