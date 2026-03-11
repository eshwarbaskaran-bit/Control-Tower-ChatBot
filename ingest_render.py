import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# ── Lightweight embeddings — matches agent_render.py ─────────────────────────
print("🧠 Loading MiniLM embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ── Load data.txt ─────────────────────────────────────────────────────────────
data_path = Path(__file__).parent / "data.txt"
print(f"📄 Loading {data_path}...")
loader = TextLoader(str(data_path), encoding="utf-8")
documents = loader.load()

# ── Semantic chunking ─────────────────────────────────────────────────────────
print("✂️  Semantic chunking...")
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
chunks = splitter.split_documents(documents)
print(f"📦 {len(chunks)} chunks created")

# ── Push to Supabase ──────────────────────────────────────────────────────────
print("🧠 Generating MiniLM embeddings → PostgreSQL...")
CONNECTION_STRING = os.environ["DATABASE_URL"]

db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="control_tower_docs_minilm",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,
    use_jsonb=True,
)

print(f"✅ Ingestion complete. {len(chunks)} semantic chunks stored in control_tower_docs_minilm.")
print("💡 If chunk quality looks off, tune breakpoint_threshold_amount (try 75–90).")