import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Only needed for the LLM reasoning now!
os.environ["GOOGLE_API_KEY"] = "AIzaSyCZGBWvQWQiYRe07QORIKfOC1CXYxQ8qmU" 

class SemanticSniperAgent:
    def __init__(self):
        # 1. Initialize Local Embeddings (MUST match ingest.py exactly)
        print("🧠 [LOCAL]: Loading HuggingFace Semantic Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # 2. Load the Semantic FAISS Database
        print("💾 [DB]: Loading Semantic Index...")
        self.vector_db = FAISS.load_local(
            "faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

        # 3. Setup Gemini 2.5 Flash for reasoning
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

        # 4. High-Precision Retriever
        # Because we have 90 semantic chunks, k=4 will pull 4 entire topics.
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 4, 'fetch_k': 20}
        )

        # 5. The Professional Architect Prompt
        system_prompt = (
            "You are the Lead Solutions Architect for ClickPost. "
            "Use the provided documentation chunks to answer the question with technical precision.\n\n"
            "RULES:\n"
            "1. Use bold headings and bullet points for logic.\n"
            "2. If a specific status code or formula is in the text, include it.\n"
            "3. If the answer isn't in the context, say you don't know.\n\n"
            "CONTEXT:\n{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 6. Build the Chain
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

    def ask(self, query):
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

if __name__ == "__main__":
    bot = SemanticSniperAgent()
    
    # Test it with a "logic" question to see the semantic power
    question = "how to find stuck revenue"
    print(f"\n👤 USER: {question}\n")
    
    answer = bot.ask(question)
    print(f"🎯 SEMANTIC RESPONSE:\n{answer}")