import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def run_elite_semantic_ingest():
    print("🗼 [ACTION]: Starting Elite Context-Aware Semantic Ingestion...")
    
    if not os.path.exists("data.txt"):
        print("❌ Error: data.txt not found.")
        return

    # 1. Load the raw text
    with open("data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("🧠 [LOCAL]: Booting up HuggingFace mpnet-base-v2...")
    # Explicitly forcing it to use CPU (avoids silent memory errors)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'} 
    )

    # --- THE ELITE UPGRADE STARTS HERE ---
    
    # 2. Structural Parsing (Extract the Metadata)
    print("🔪 [STEP 1]: Parsing Markdown Structure...")
    headers_to_split_on = [
        ("#", "Module"),
        ("##", "Widget Name"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = markdown_splitter.split_text(raw_text)

    # 3. Context Injection (The Secret Sauce)
    # We force the Widget Name directly into the text string so the embedding math reads it.
    enriched_docs = []
    for doc in md_docs:
        module_name = doc.metadata.get("Module", "Control Tower")
        widget_name = doc.metadata.get("Widget Name", "General Logic")
        
        # Create a physical tag at the top of the text
        context_tag = f"[CONTEXT: {module_name} -> {widget_name}]\n"
        enriched_content = context_tag + doc.page_content
        
        enriched_docs.append(Document(page_content=enriched_content, metadata=doc.metadata))

    print(f"✅ Extracted {len(enriched_docs)} context-tagged widgets.")

    # 4. Deep Semantic Slicing
    print("🔪 [STEP 2]: Deep Semantic Slicing within Widgets...")
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85 # Tuned to 85% to prevent over-shredding
    )
    
    # Split the enriched documents instead of the raw text
    final_chunks = semantic_splitter.split_documents(enriched_docs)
    print(f"📦 [INFO]: Refined into {len(final_chunks)} context-aware semantic chunks.")

    # 5. Save to FAISS
    print("💾 [INFO]: Saving Elite Index to local Vector DB...")
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        
    vector_db = FAISS.from_documents(final_chunks, embeddings)
    vector_db.save_local("faiss_index")
    
    print("✨ [SUCCESS]: Elite Semantic Database built and saved locally!")

if __name__ == "__main__":
    run_elite_semantic_ingest()
