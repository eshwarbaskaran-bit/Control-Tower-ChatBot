"""
graph/nodes.py
──────────────
Node functions and conditional-edge routers for the Control Tower LangGraph agent.

Each node:
    - Receives full GraphState
    - Does ONE job (single-responsibility)
    - Returns a partial dict with only the keys it updates

Routing functions:
    - Return a string key that maps to the next node in the graph
    - Used by add_conditional_edges() in agent_graph.py
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from graph.state import GraphState
from graph.config import settings
from graph.prompts import (
    ANALYZER_PROMPT,
    GRADER_PROMPT,
    REWRITER_PROMPT,
    GENERATOR_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Strip markdown fences and whitespace from LLM JSON output.

    Llama 3.3 on Groq occasionally wraps JSON in ```json ... ``` blocks
    or adds a trailing explanation. This handles both.
    """
    text = text.strip()
    # Remove ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # If there's text after the closing brace, drop it
    brace_end = text.rfind("}")
    if brace_end != -1:
        text = text[: brace_end + 1]
    return text.strip()


def _safe_parse_json(text: str, fallback: dict) -> dict:
    """Parse JSON from LLM output with graceful fallback.

    Args:
        text: Raw LLM response string.
        fallback: Dict to return if parsing fails.

    Returns:
        Parsed dict or fallback.
    """
    try:
        return json.loads(_clean_json(text))
    except (json.JSONDecodeError, ValueError):
        print(f"[nodes] ⚠️ JSON parse failed. Raw text: {text[:200]}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Analyze query
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_query(state: GraphState, *, llm: Any) -> dict:
    """Classify intent, extract entities, and detect ambiguity.

    LLM calls: 1
    Sets: intent, entities, needs_clarification, clarification_question
    """
    chain = ANALYZER_PROMPT | llm

    response = await chain.ainvoke({
        "input": state["input"],
        "chat_history": state.get("chat_history", []),
    })

    parsed = _safe_parse_json(
        response.content,
        fallback={
            "intent": "definition",
            "entities": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
    )

    return {
        "intent": parsed.get("intent", "definition"),
        "entities": parsed.get("entities", []),
        "needs_clarification": parsed.get("needs_clarification", False),
        "clarification_question": parsed.get("clarification_question"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Retrieve documents
# ─────────────────────────────────────────────────────────────────────────────

async def retrieve(state: GraphState, *, retriever: Any) -> dict:
    """Fetch documents from PGVector via MMR.

    Uses query_rewrite if available (from a previous rewrite loop),
    otherwise uses the raw input. Appends extracted entities for
    better recall.

    LLM calls: 0
    Sets: documents
    """
    # Build the search query
    base_query = state.get("query_rewrite") or state["input"]

    # Append entities for better recall (if analyzer extracted any)
    entities = state.get("entities", [])
    if entities:
        entity_suffix = " " + " ".join(entities)
        search_query = base_query + entity_suffix
    else:
        search_query = base_query

    print(f"[retrieve] 🔍 Searching: {search_query}")

    documents = await retriever.ainvoke(search_query)

    print(f"[retrieve] 📦 Got {len(documents)} documents")
    for i, doc in enumerate(documents):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] {preview}...")

    return {"documents": documents}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Grade document relevance
# ─────────────────────────────────────────────────────────────────────────────

async def _grade_single_doc(
    doc: Document, question: str, llm: Any
) -> tuple[Document, bool]:
    """Grade a single document for relevance. Returns (doc, is_relevant)."""
    chain = GRADER_PROMPT | llm

    response = await chain.ainvoke({
        "input": question,
        "document_content": doc.page_content,
    })

    parsed = _safe_parse_json(
        response.content,
        fallback={"relevant": "yes"},  # default to keeping doc on parse failure
    )

    is_relevant = parsed.get("relevant", "yes").lower().strip() == "yes"
    return doc, is_relevant


async def grade_documents(state: GraphState, *, llm: Any) -> dict:
    """Score each retrieved document for relevance, in parallel.

    Documents scored "no" are dropped. The surviving subset becomes
    filtered_docs for the generator.

    LLM calls: len(documents)  (parallel via asyncio.gather)
    Sets: filtered_docs
    """
    documents = state.get("documents", [])
    question = state["input"]

    if not documents:
        print("[grade] ⚠️ No documents to grade")
        return {"filtered_docs": []}

    # Grade all docs in parallel
    tasks = [_grade_single_doc(doc, question, llm) for doc in documents]
    results = await asyncio.gather(*tasks)

    filtered = [doc for doc, is_relevant in results if is_relevant]
    dropped = [doc for doc, is_relevant in results if not is_relevant]

    print(
        f"[grade] ✅ {len(filtered)}/{len(documents)} documents passed relevance check"
    )
    for doc in dropped:
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [dropped] {preview}...")

    return {"filtered_docs": filtered}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Rewrite query
# ─────────────────────────────────────────────────────────────────────────────

async def rewrite_query(state: GraphState, *, llm: Any) -> dict:
    """Reformulate the search query for better retrieval on retry.

    LLM calls: 1
    Sets: query_rewrite, retry_count
    """
    chain = REWRITER_PROMPT | llm

    current_retry = state.get("retry_count", 0)

    response = await chain.ainvoke({
        "input": state["input"],
        "intent": state.get("intent", "definition"),
        "retry_count": current_retry + 1,
    })

    rewritten = response.content.strip().strip('"').strip("'")

    print(f"[rewrite] ✏️ Retry {current_retry + 1}: {rewritten}")

    return {
        "query_rewrite": rewritten,
        "retry_count": current_retry + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: Generate answer
# ─────────────────────────────────────────────────────────────────────────────

async def generate_answer(state: GraphState, *, llm: Any) -> dict:
    """Produce the final answer from graded documents + chat history.

    LLM calls: 1
    Sets: answer
    """
    filtered_docs = state.get("filtered_docs", [])
    intent = state.get("intent", "definition")

    # Build context string from filtered docs
    if filtered_docs:
        context = "\n\n---\n\n".join(
            doc.page_content for doc in filtered_docs
        )
    else:
        context = "(No relevant documents found in knowledge base.)"

    chain = GENERATOR_PROMPT | llm

    response = await chain.ainvoke({
        "input": state["input"],
        "intent": intent,
        "context": context,
        "chat_history": state.get("chat_history", []),
    })

    return {"answer": response.content}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER: After analysis
# ─────────────────────────────────────────────────────────────────────────────

def route_after_analysis(state: GraphState) -> str:
    """Decide where to go after the analyzer runs.

    Returns:
        "clarify"   → needs_clarification is True → END (return question)
        "off_topic" → intent is off_topic → generator (polite decline)
        "retrieve"  → everything else → normal retrieval path
    """
    if state.get("needs_clarification", False):
        return "clarify"

    if state.get("intent") == "off_topic":
        return "off_topic"

    return "retrieve"


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER: After grading
# ─────────────────────────────────────────────────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    """Decide whether to rewrite or generate after grading.

    Returns:
        "rewrite"  → too few relevant docs AND retries remaining
        "generate" → enough docs OR retries exhausted
    """
    filtered_docs = state.get("filtered_docs", [])
    retry_count = state.get("retry_count", 0)

    has_enough_docs = len(filtered_docs) >= settings.MIN_RELEVANT_DOCS
    can_retry = retry_count < settings.MAX_REWRITE_RETRIES

    if not has_enough_docs and can_retry:
        print(
            f"[route] 🔄 Not enough docs ({len(filtered_docs)}/{settings.MIN_RELEVANT_DOCS}), "
            f"retry {retry_count + 1}/{settings.MAX_REWRITE_RETRIES}"
        )
        return "rewrite"

    if not has_enough_docs:
        print(
            f"[route] ⚠️ Not enough docs and retries exhausted. "
            f"Generating with {len(filtered_docs)} docs."
        )

    return "generate"