"""
graph/agent_graph.py
────────────────────
Compiles the Control Tower LangGraph agent.

Initializes LLM, embeddings, and PGVector once at startup.
Binds resources to node functions and wires the graph edges.

Usage
─────
    from graph.agent_graph import build_graph

    # In Chainlit on_chat_start:
    graph = build_graph(is_async=True)
    result = await graph.ainvoke({
        "input": "What is pending pickup?",
        "chat_history": [],
        "retry_count": 0,
        "needs_clarification": False,
    })
    print(result["answer"])
"""

from __future__ import annotations

from functools import partial

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

from graph.state import GraphState
from graph.config import settings
from graph.nodes import (
    analyze_query,
    retrieve,
    grade_documents,
    rewrite_query,
    generate_answer,
    route_after_analysis,
    route_after_grading,
)


# ─────────────────────────────────────────────────────────────────────────────
# Clarification handler — bridges the "ambiguous" path to END
# ─────────────────────────────────────────────────────────────────────────────

async def handle_clarification(state: GraphState) -> dict:
    """Copy clarification_question into answer so main_sb.py can read it.

    This node exists because LangGraph's END doesn't allow state updates.
    When the analyzer flags ambiguity, the graph routes here, sets answer,
    then proceeds to END.
    """
    return {
        "answer": state.get("clarification_question", "Could you rephrase that?"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(is_async: bool = True):
    """Build and compile the Control Tower LangGraph agent.

    Args:
        is_async: If True, uses async SQLAlchemy engine for PGVector.
                  Always True when called from Chainlit (main_sb.py).
                  Set False for sync testing scripts.

    Returns:
        Compiled LangGraph (CompiledStateGraph) ready for .ainvoke() or .invoke().
    """

    # ── 1. Initialize LLM ───────────────────────────────────────────────
    print("🧠 [Graph] Initializing Groq LLM...")
    llm = ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_retries=settings.LLM_MAX_RETRIES,
        api_key=settings.GROQ_API_KEY,
    )

    # ── 2. Initialize embeddings ─────────────────────────────────────────
    print(f"🧠 [Graph] Loading embeddings: {settings.EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── 3. Initialize PGVector ───────────────────────────────────────────
    print(f"💾 [Graph] Connecting to PostgreSQL (async={is_async})...")
    connection_string = settings.BOT_DB_URL

    if is_async:
        # Swap driver prefix for asyncpg
        if not connection_string.startswith("postgresql+"):
            connection_string = connection_string.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        elif "+psycopg" in connection_string:
            connection_string = connection_string.replace(
                "+psycopg", "+asyncpg"
            )
        engine = create_async_engine(connection_string)
    else:
        if "+asyncpg" in connection_string:
            connection_string = connection_string.replace("+asyncpg", "")
        if "+psycopg" in connection_string:
            connection_string = connection_string.replace("+psycopg", "")
        engine = create_engine(connection_string)

    vector_db = PGVector(
        embeddings=embeddings,
        collection_name=settings.COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
        create_extension=False,
    )

    retriever = vector_db.as_retriever(
        search_type=settings.SEARCH_TYPE,
        search_kwargs=settings.retriever_kwargs,
    )

    # ── 4. Bind resources to node functions ──────────────────────────────
    # functools.partial pre-fills the llm/retriever kwargs so each node
    # conforms to LangGraph's (state) -> dict signature.
    analyze_node = partial(analyze_query, llm=llm)
    retrieve_node = partial(retrieve, retriever=retriever)
    grade_node = partial(grade_documents, llm=llm)
    rewrite_node = partial(rewrite_query, llm=llm)
    generate_node = partial(generate_answer, llm=llm)

    # ── 5. Build the graph ───────────────────────────────────────────────
    print("🔗 [Graph] Wiring LangGraph nodes...")
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("clarify", handle_clarification)

    # Entry point
    graph.set_entry_point("analyze")

    # After analysis: route to clarify, generate (off-topic), or retrieve
    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "clarify": "clarify",       # ambiguous → ask user
            "off_topic": "generate",    # off-topic → polite decline
            "retrieve": "retrieve",     # normal → vector search
        },
    )

    # Clarify → END
    graph.add_edge("clarify", END)

    # Retrieve → Grade
    graph.add_edge("retrieve", "grade")

    # After grading: route to rewrite or generate
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "rewrite": "rewrite",       # not enough docs → reformulate
            "generate": "generate",     # enough docs → answer
        },
    )

    # Rewrite → Retrieve (loop back)
    graph.add_edge("rewrite", "retrieve")

    # Generate → END
    graph.add_edge("generate", END)

    # ── 6. Compile ───────────────────────────────────────────────────────
    compiled = graph.compile()
    print("✅ [Graph] Control Tower agent compiled and ready.")

    return compiled