import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise EnvironmentError("❌ DATABASE_URL not set. Copy .env.template → .env first.")

CONNECTION_STRING = os.environ["DATABASE_URL"]

# ── Embedding model ───────────────────────────────────────────────────────────
# MUST match agent.py — if you change this, re-ingest everything
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


def migrate_to_postgres():
    print("🚀 Booting up Postgres Ingestion with Semantic Chunking...")

    # ── Load raw docs ─────────────────────────────────────────────────────────
    loader = TextLoader("data.txt", encoding="utf-8")
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} document(s).")

    # ── Semantic Chunker ──────────────────────────────────────────────────────
    # How it works:
    #   1. Splits the text into sentences first.
    #   2. Embeds each sentence using the SAME embedding model as retrieval.
    #   3. Computes cosine similarity between consecutive sentences.
    #   4. When similarity drops sharply (topic shift), it starts a new chunk.
    #   Result: each chunk is topically coherent — "Pending Pickup" logic won't
    #   bleed into "Stuck at Hub" logic just because of character limits.
    #
    # breakpoint_threshold_type options:
    #   "percentile"         — split when similarity drops below the Nth percentile
    #                          of all sentence-pair similarities in the document.
    #                          Best default for structured KB docs.
    #   "standard_deviation" — split when drop exceeds 1 std dev below mean.
    #                          Better for very long, varied documents.
    #   "interquartile"      — uses IQR, more robust to outliers.
    #
    # For a structured logistics KB, "percentile" at 85 gives clean concept-level
    # chunks. Tune breakpoint_threshold_amount after inspecting the sample output.
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,  # higher = fewer, larger chunks
                                          # lower  = more, tighter chunks
    )

    chunks = text_splitter.split_documents(docs)

    print(f"📦 Semantic chunker produced {len(chunks)} chunks.")

    # ── Sanity check: print first 3 chunks so you can verify quality ─────────
    print("\n── Sample Chunks (first 3) ──────────────────────────────────────")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}] ({len(chunk.page_content)} chars)")
        print(chunk.page_content[:300])
        print("...")
    print("─────────────────────────────────────────────────────────────────\n")

    print("🧠 Generating BGE embeddings → PostgreSQL...")

    PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name="control_tower_docs",
        connection=CONNECTION_STRING,
        use_jsonb=True,
        pre_delete_collection=True,  # wipe old vectors before re-ingesting
    )

    print(f"✅ Ingestion complete. {len(chunks)} semantic chunks stored.")
    print("💡 If chunk quality looks off, tune breakpoint_threshold_amount (try 75–90).")


if __name__ == "__main__":
    migrate_to_postgres()