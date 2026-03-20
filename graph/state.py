"""
graph/state.py
──────────────
LangGraph state schema for the Control Tower chatbot.

Every node receives the full state and returns a *partial* dict
containing only the keys it wants to update. LangGraph merges
the update into the running state automatically.

Design notes
────────────
- chat_history is injected by the Chainlit layer (main_sb.py)
  before each graph invocation. The graph reads it but never
  manages its lifecycle (appending, trimming, persisting).

- retry_count exists to cap the retrieve → grade → rewrite loop.
  Max 2 retries. After that the generator works with whatever
  filtered_docs it has (possibly empty → "not in my knowledge base").

- needs_clarification is the analyzer's escape hatch. When True
  the graph routes straight to END and returns clarification_question
  as the response. Chainlit skips feedback buttons in this case.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """Full state flowing through the Control Tower LangGraph agent."""

    # ── Inputs (set once before graph starts) ────────────────────────────
    input: str
    """Raw user message, never modified by any node."""

    chat_history: list[BaseMessage]
    """Sliding window of recent conversation turns.
    Injected by main_sb.py from Chainlit's user_session.
    Read by: analyzer (to understand follow-ups), generator (for context).
    """

    # ── Analyzer outputs ─────────────────────────────────────────────────
    intent: str | None
    """Classified intent of the user query.
    One of: definition, navigation, alert_setup, comparison, off_topic, ambiguous.
    Set by: analyze_query node.
    Read by: route_after_analysis (conditional edge), generator (formatting).
    """

    entities: list[str]
    """Key entities extracted from the query.
    Examples: ['forward movement', 'stuck shipments'], ['RTO dashboard'].
    Set by: analyze_query node.
    Read by: retrieve node (appended to search query for better recall).
    """

    needs_clarification: bool
    """True when the analyzer decides the query is too vague to retrieve on.
    Set by: analyze_query node.
    Read by: route_after_analysis (conditional edge) → short-circuits to END.
    """

    clarification_question: str | None
    """The question to ask the user when needs_clarification is True.
    Example: 'Which dashboard are you asking about — Forward, Reverse, or RTO?'
    Set by: analyze_query node.
    Read by: main_sb.py (sent as the bot response instead of an answer).
    """

    # ── Retriever outputs ────────────────────────────────────────────────
    documents: list[Document]
    """Raw documents from PGVector MMR retrieval (k=6).
    Set by: retrieve node.
    Read by: grade_documents node.
    """

    # ── Grader outputs ───────────────────────────────────────────────────
    filtered_docs: list[Document]
    """Documents that passed relevance grading.
    Subset of documents — could be 0 to k items.
    Set by: grade_documents node.
    Read by: route_after_grading (conditional edge), generator.
    """

    # ── Rewrite loop ─────────────────────────────────────────────────────
    query_rewrite: str | None
    """Reformulated search query after a failed retrieval pass.
    Set by: rewrite_query node.
    Read by: retrieve node (uses this instead of raw input if present).
    """

    retry_count: int
    """Number of retrieve→grade→rewrite cycles completed.
    Starts at 0. Max 2 — after that, generator uses whatever it has.
    Set by: rewrite_query node (increments by 1).
    Read by: route_after_grading (conditional edge).
    """

    # ── Generator output ─────────────────────────────────────────────────
    answer: str
    """Final response string sent to the user.
    Set by: generate node.
    Read by: main_sb.py (streamed to Chainlit).
    """