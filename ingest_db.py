import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. The Database Connection String
# Format: postgresql+psycopg://username:password@host:port/database_name
CONNECTION_STRING = "postgresql+psycopg://admin:clickpost123@localhost:5432/ct_copilot"

# 2. Define the exact same Embedding Model you used before
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def migrate_to_postgres():
    print("🚀 Booting up Postgres Ingestion...")

    # 3. Load your documentation (Make sure data.txt is in your folder!)
    loader = TextLoader("data.txt", encoding="utf-8")
    docs = loader.load()

    # 4. Chunk the text exactly like we did for FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    print(f"📦 Extracted {len(chunks)} chunks from the documentation.")
    print("🧠 Generating embeddings and writing to PostgreSQL. This might take a minute...")

    # 5. Push directly to PostgreSQL
    # This automatically creates the tables and vector indexes if they don't exist
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name="control_tower_docs",
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    print("✅ Migration Complete! Your AI brain is now running on PostgreSQL.")

if __name__ == "__main__":
    migrate_to_postgres()