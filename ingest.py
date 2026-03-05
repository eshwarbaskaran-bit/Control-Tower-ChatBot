import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS

def run_true_semantic_local_ingest():
    print("🗼 [ACTION]: Starting True Semantic Ingestion (100% Local)...")
    
    if not os.path.exists("data.txt"):
        print("❌ Error: data.txt not found.")
        return

    # 1. Load the raw text
    loader = TextLoader("data.txt", encoding="utf-8")
    documents = loader.load()

    # 2. Initialize Local HuggingFace Embeddings
    print("🧠 [LOCAL]: Booting up HuggingFace mpnet-base-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # 3. The True Semantic Chunker
    print("🔪 [CHUNK]: Analyzing sentence meanings to find natural breakpoints...")
    print("⏳ (This will calculate the vector distance of every sentence locally. Please wait...)")
    
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" # Splits only when the AI detects a topic change
    )
    
    # This processes all 1,800+ sentences locally. No API limits!
    chunks = semantic_splitter.split_documents(documents)
    print(f"📦 [INFO]: Created {len(chunks)} purely semantic chunks based on AI meaning.")

    # 4. Save to FAISS
    print("💾 [INFO]: Saving chunks to local Vector DB...")
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    
    print("✨ [SUCCESS]: True Semantic Database built and saved locally!")

if __name__ == "__main__":
    run_true_semantic_local_ingest()