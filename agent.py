import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

# 1. Load secrets securely
load_dotenv()

class SemanticSniperAgent:
    def __init__(self, is_async=False):
        print("🧠 [LOCAL]: Loading HuggingFace Semantic Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        print(f"💾 [DB]: Connecting to PostgreSQL... (Async Mode: {is_async})")
        CONNECTION_STRING = "postgresql+psycopg://admin:clickpost123@localhost:5432/ct_copilot"
        
        # 2. THE FIX: Choose the right engine based on the environment
        if is_async:
            self.engine = create_async_engine(CONNECTION_STRING)
        else:
            self.engine = create_engine(CONNECTION_STRING)
            
        self.vector_db = PGVector(
            embeddings=self.embeddings,
            collection_name="control_tower_docs",
            connection=self.engine,
            use_jsonb=True,
        )

        # 3. Setup LLM & Retriever
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 4, 'fetch_k': 20}
        )

        # 4. The Professional Architect Prompt
        system_prompt = (
            "# Role\n"
            "- You are a Senior Product Operations Analyst working at ClickPost.\n"
            "- Your objective is to explain Control Tower and logistics logic with extreme brevity.\n"
            "- Bridge the gap between system logic and physical operations using short, punchy statements.\n\n"
            "---\n\n"
            "# Guiding Principle\n"
            "Synthesize the context into a maximum of 3 to 4 concise bullet points. Never regurgitate raw text. Combine the 'What', 'Why', and 'Action' seamlessly into natural sentences. Do not use introductory fluff or output bolded subheadings.\n\n"
            "---\n\n"
            "<GUARDRAILS & STRICT BOUNDARIES>\n"
            "1. ZERO HALLUCINATION: Only use the provided `CONTEXT`. Do not invent features or logic.\n"
            "2. NO FORMATTING FLUFF: NEVER output headings like 'Underlying Logic', 'Physical Reality', or 'Action'. Output ONLY a single list of 3-4 bullet points.\n"
            "3. OUT OF SCOPE REJECTION: Refuse to answer anything unrelated to ClickPost or logistics.\n"
            "</GUARDRAILS & STRICT BOUNDARIES>\n\n"
            "---\n\n"
            "CONTEXT:\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 5. Build the Chain
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

    # Synchronous call for Terminal/FastAPI
    def ask(self, query):
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

    # Asynchronous call for Chainlit
    async def aask(self, query):
        response = await self.retrieval_chain.ainvoke({"input": query})
        return response["answer"]

# Terminal test
if __name__ == "__main__":
    bot = SemanticSniperAgent(is_async=False)
    question = "What is pending pickup?"
    print(f"\n👤 USER: {question}\n")
    answer = bot.ask(question)
    print(f"🎯 SEMANTIC RESPONSE:\n{answer}")