"""
ingest_render.py
────────────────
Ingests data.txt into PGVector using the custom structure-aware splitter.

Replaces semantic chunking with marker-based splitting so each widget entry,
how-to block, and worked example stays as one atomic chunk.

Usage:
    python ingest_render.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from graph.splitter import split_control_tower_docs

# ── Lightweight embeddings — matches agent_graph.py ──────────────────────────
print("🧠 Loading MiniLM embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ── Load data.txt ─────────────────────────────────────────────────────────────
data_path = Path(__file__).parent / "data.txt"
print(f"📄 Loading {data_path}...")
text = data_path.read_text(encoding="utf-8")

# ── Structure-aware chunking ─────────────────────────────────────────────────
print("✂️  Splitting by document structure (sections, widgets, how-tos)...")
chunks = split_control_tower_docs(text)
print(f"📦 {len(chunks)} chunks created")

# Show chunk breakdown by type
from collections import Counter
type_counts = Counter(c.metadata["block_type"] for c in chunks)
for btype, count in sorted(type_counts.items()):
    print(f"    {btype}: {count}")

# Preview first few chunks
print("\n📋 First 5 chunks preview:")
for i, chunk in enumerate(chunks[:5]):
    meta = chunk.metadata
    preview = chunk.page_content[:100].replace("\n", " ")
    print(f"  [{i+1}] [{meta['block_type']}] {meta['block_title']}")
    print(f"       {preview}...")

# ── Push to database ──────────────────────────────────────────────────────────
print("\n🧠 Generating MiniLM embeddings → PostgreSQL...")
CONNECTION_STRING = os.environ["BOT_DB_URL"]

db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="control_tower_docs_minilm",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
    use_jsonb=True,
)

print(f"\n✅ Ingestion complete. {len(chunks)} structured chunks stored in control_tower_docs_minilm.")