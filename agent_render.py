import os
from dotenv import load_dotenv
from pathlib import Path

# ─────────────────────────────────────────────
# LangSmith — must be set BEFORE any LangChain import
# ─────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── Validate required keys early — fail loud, not silently ───────────────────
_REQUIRED = ["GROQ_API_KEY", "BOT_DB_URL"]
_missing = [k for k in _REQUIRED if not os.getenv(k)]
if _missing:
    raise EnvironmentError(
        f"❌ Missing required environment variables: {', '.join(_missing)}\n"
        "Please fill in all values in your .env file."
    )

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "control-tower-bot")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine


class SemanticSniperAgent:
    def __init__(self, is_async=False):
        print("🧠 [RENDER]: Loading lightweight MiniLM embeddings...")

        # ── Lightweight model — fits in 512MB Render free tier ───────────────
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"💾 [DB]: Connecting to PostgreSQL... (Async: {is_async})")
        CONNECTION_STRING = os.environ["BOT_DB_URL"]

        if is_async:
            if not CONNECTION_STRING.startswith("postgresql+"):
                CONNECTION_STRING = CONNECTION_STRING.replace("postgresql://", "postgresql+asyncpg://")
            self.engine = create_async_engine(CONNECTION_STRING)
        else:
            if "+asyncpg" in CONNECTION_STRING:
                CONNECTION_STRING = CONNECTION_STRING.replace("+asyncpg", "")
            self.engine = create_engine(CONNECTION_STRING)

        self.vector_db = PGVector(
            embeddings=self.embeddings,
            collection_name="control_tower_docs_minilm",
            connection=self.engine,
            use_jsonb=True,
            create_extension=False,
        )

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_retries=3,
            api_key=os.environ["GROQ_API_KEY"],
        )

        # ── Retriever ────────────────────────────────────────────────────────
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.6},
        )

        # ── Prompt ───────────────────────────────────────────────────────────
        system_prompt = """
# Role
You are a Senior Product Operations Analyst at ClickPost with deep expertise in Control Tower.

# Dashboard Routing (use ONLY if context confirms the dashboard name)
- Q-Commerce / Rider / ETA / Dark Store → Quick Commerce dashboard
- Reverse / RTO / Return               → Reverse Movement dashboard  
- Forward / Delivery / PDD / Stuck     → Forward Movement dashboard
If context gives a different name, use the context name — not this table.

# Response Style
- DEFINITIONS ("What is X?"): 3–4 bullets, logic + business impact.
- NAVIGATION ("How do I check X?"): Numbered steps, EXACT widget/dashboard names from CONTEXT only.
- ALERTS ("How do I set up alert for X?"): Extract all config fields from CONTEXT. Combine across chunks intelligently. Never use placeholders.
- COMPARISONS: Answer directly, clarify nuances.

# Hard Rules
1. NEVER invent info not in CONTEXT. Say "not in my knowledge base" if missing.
2. NEVER output placeholders — use real values or admit you don't have them.
3. No preamble. Answer on line 1.
4. Refuse non-ClickPost questions.

---
CONTEXT:
{context}
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

    def ask(self, query: str) -> str:
        response = self.retrieval_chain.invoke({"input": query})
        return response["answer"]

    async def aask(self, query: str) -> str:
        response = await self.retrieval_chain.ainvoke({"input": query})
        return response["answer"]


if __name__ == "__main__":
    bot = SemanticSniperAgent(is_async=False)
    question = "What is pending pickup?"
    print(f"\n👤 USER: {question}\n")
    answer = bot.ask(question)
    print(f"🎯 RESPONSE:\n{answer}")