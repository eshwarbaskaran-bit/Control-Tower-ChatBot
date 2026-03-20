"""
main_lg.py
──────────
LangGraph version of the Chainlit app — auth + feedback + CSV upload + LangGraph agent.

Runs alongside main_sb.py (original) without touching it.
Once validated, this replaces main_sb.py in production.

Launch:  chainlit run main_lg.py
Ngrok:   ngrok http --url=shrieval-blondell-longly.ngrok-free.dev 8000
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import asyncpg
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

from graph.agent_graph import build_graph
from graph.config import settings
from graph.csv_handler import parse_csv, resolve_widget_name, query_widget, get_data_summary


# ─────────────────────────────────────────────────────────────────────────────
# Auth — single shared login for internal team
# ─────────────────────────────────────────────────────────────────────────────
USERNAME = "clickpost"
PASSWORD = "ct2025"


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == USERNAME and password == PASSWORD:
        return cl.User(identifier=username, metadata={"role": "internal"})
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Supabase feedback writer (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
async def write_feedback(rating: str, comment: str | None):
    raw_url = settings.SUPABASE_FEEDBACK_URL
    if not raw_url:
        print("[Feedback] ⚠️ SUPABASE_FEEDBACK_URL not set, skipping write")
        return

    if "://" in raw_url:
        pg_url = "postgresql://" + raw_url.split("://", 1)[1]
    else:
        pg_url = raw_url
    pg_url = pg_url.replace("?sslmode=require", "").replace("&sslmode=require", "")

    try:
        conn = await asyncpg.connect(pg_url, ssl="require")
        await conn.execute(
            """
            INSERT INTO ct_feedback
                (created_at, username, question, answer, rating, comment, session_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            datetime.now(timezone.utc),
            cl.user_session.get("username", "clickpost"),
            cl.user_session.get("last_question", ""),
            cl.user_session.get("last_answer", ""),
            rating,
            comment,
            cl.context.session.id,
        )
        await conn.close()
        print(f"[Feedback] ✅ Written to Supabase: {rating}")
    except Exception as e:
        print(f"[Feedback] DB write failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Format data query results into a readable message
# ─────────────────────────────────────────────────────────────────────────────
def format_data_result(result: dict) -> str:
    """Format a widget query result into a human-readable message."""
    lines = []

    lines.append(f"**{result['widget_name']}** — {result['description']}")
    lines.append(f"**Total shipments:** {result['total_count']}")

    if "time_buckets" in result:
        lines.append("\n**Time bucket breakdown:**")
        for bucket, count in result["time_buckets"].items():
            lines.append(f"  • {bucket}: {count} shipments")

    if "carrier_breakdown" in result:
        lines.append("\n**Top carriers:**")
        for carrier, count in result["carrier_breakdown"].items():
            lines.append(f"  • {carrier}: {count}")

    if "grouped_by" in result:
        group = result["grouped_by"]
        lines.append(f"\n**Breakdown by {group['column']}:**")
        for label, count in group["breakdown"].items():
            lines.append(f"  • {label}: {count}")

    # Shipment table
    if "shipment_table" in result and result["shipment_table"]:
        shipments = result["shipment_table"]

        if result.get("table_truncated"):
            lines.append(f"\n**Shipments (showing 50 of {result['table_total']}):**")
        else:
            lines.append(f"\n**Shipments:**")

        # Build markdown table
        headers = list(shipments[0].keys())
        # Clean up header names for display
        display_headers = [h.replace("_", " ").title() for h in headers]

        header_row = "| " + " | ".join(display_headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines.append(header_row)
        lines.append(separator)

        for row in shipments:
            values = [str(row.get(h, "-")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chat Start
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="🏗️ Booting ClickPost Logic Engine...")
    await msg.send()

    try:
        graph = build_graph(is_async=True)

        cl.user_session.set("graph", graph)
        cl.user_session.set("chat_history", [])
        cl.user_session.set("last_question", "")
        cl.user_session.set("last_answer", "")
        cl.user_session.set("awaiting_feedback", False)
        cl.user_session.set("shipment_data", None)  # DataFrame storage

        user = cl.context.session.user
        cl.user_session.set("username", user.identifier if user else "clickpost")

        msg.content = "✅ **Control Tower Online.** Ask me anything about ClickPost logistics.\n\n📎 You can also upload a CSV export to analyze your shipment data."
        await msg.update()

    except Exception as e:
        msg.content = f"❌ **Startup Error:** {str(e)}"
        await msg.update()


# ─────────────────────────────────────────────────────────────────────────────
# On Message
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):

    # ── Intercept free-text feedback ─────────────────────────────────────
    if cl.user_session.get("awaiting_feedback", False):
        cl.user_session.set("awaiting_feedback", False)
        await write_feedback(rating="negative", comment=message.content)
        await cl.Message(content="📝 Got it — feedback saved. Thanks!").send()
        return

    # ── Handle CSV file upload ───────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path and element.path.endswith(".csv"):
                await _handle_csv_upload(element)
                return
            elif hasattr(element, "name") and element.name and element.name.endswith(".csv"):
                await _handle_csv_upload(element)
                return

        # Non-CSV file uploaded
        await cl.Message(
            content="I can only process CSV files right now. Please upload a .csv export from ClickPost."
        ).send()
        return

    # ── Check if this is a data query (CSV loaded + widget-matchable question) ──
    df = cl.user_session.get("shipment_data")
    widget_key = resolve_widget_name(message.content) if df is not None else None

    if df is not None and widget_key:
        await _handle_data_query(message, df, widget_key)
        return

    # ── Run the LangGraph agent (knowledge base query) ───────────────────
    await _handle_graph_query(message)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Upload Handler
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_csv_upload(element):
    """Parse and store a CSV file from the user."""
    msg = cl.Message(content="📊 Parsing your CSV...")
    await msg.send()

    try:
        # Read the file
        if hasattr(element, "path") and element.path:
            with open(element.path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            content = element.content

        df, summary = parse_csv(content)
        cl.user_session.set("shipment_data", df)

        # Build response
        summary_text = get_data_summary(df)
        response = f"✅ **CSV loaded successfully!**\n\n{summary_text}"
        response += "\n\nYou can now ask me things like:"
        response += "\n• *Show me pending pickups*"
        response += "\n• *How many shipments are stuck at destination hub?*"
        response += "\n• *Show me failed deliveries*"
        response += "\n• *What are my RTO marked shipments?*"

        msg.content = response
        await msg.update()

        print(f"[CSV] ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    except Exception as e:
        msg.content = f"❌ **CSV parsing failed:** {str(e)}\n\nMake sure this is a tab-separated ClickPost export."
        await msg.update()
        print(f"[CSV] ❌ Parse error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Data Query Handler (CSV + Knowledge Base hybrid)
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_data_query(message: cl.Message, df, widget_key: str):
    """Answer a question using both uploaded data and the knowledge base."""
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Query the uploaded data
        data_result = query_widget(df, widget_key)
        data_text = format_data_result(data_result)

        # Also run the graph for knowledge base context (widget definition + navigation)
        graph = cl.user_session.get("graph")
        chat_history = cl.user_session.get("chat_history", [])

        graph_result = await graph.ainvoke(
            {
                "input": message.content,
                "chat_history": chat_history,
                "retry_count": 0,
                "needs_clarification": False,
            },
            config={
                "tags": ["control-tower", "langgraph", "data-query"],
                "metadata": {
                    "session_id": cl.context.session.id,
                    "user_query": message.content,
                    "widget_key": widget_key,
                },
            },
        )

        kb_answer = graph_result.get("answer", "")
        is_clarification = graph_result.get("needs_clarification", False)

        # Combine: data results first, then KB context
        if is_clarification:
            answer = data_text
        else:
            answer = f"📊 **From your uploaded data:**\n\n{data_text}"
            answer += f"\n\n---\n\n📖 **From knowledge base:**\n\n{kb_answer}"

    except Exception as e:
        answer = f"⚠️ Error analyzing data: {str(e)}"

    msg.content = answer
    await msg.update()

    # Update history and feedback tracking
    cl.user_session.set("last_question", message.content)
    cl.user_session.set("last_answer", answer)

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=answer))
    max_messages = settings.MEMORY_WINDOW * 2
    if len(chat_history) > max_messages:
        chat_history = chat_history[-max_messages:]
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(
        content="Was this helpful?",
        actions=[
            cl.Action(
                name="feedback_positive",
                payload={"rating": "positive"},
                label="👍  Yes",
            ),
            cl.Action(
                name="feedback_negative",
                payload={"rating": "negative"},
                label="👎  No",
            ),
        ],
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Graph Query Handler (knowledge base only)
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_graph_query(message: cl.Message):
    """Answer a question using the LangGraph knowledge base agent."""
    graph = cl.user_session.get("graph")
    chat_history = cl.user_session.get("chat_history", [])

    msg = cl.Message(content="")
    await msg.send()

    try:
        result = await graph.ainvoke(
            {
                "input": message.content,
                "chat_history": chat_history,
                "retry_count": 0,
                "needs_clarification": False,
            },
            config={
                "tags": ["control-tower", "langgraph"],
                "metadata": {
                    "session_id": cl.context.session.id,
                    "user_query": message.content,
                },
            },
        )

        answer = result.get("answer", "⚠️ No answer generated.")
        is_clarification = result.get("needs_clarification", False)

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"
        is_clarification = False

    msg.content = answer
    await msg.update()

    # Update chat history
    if not is_clarification:
        chat_history.append(HumanMessage(content=message.content))
        chat_history.append(AIMessage(content=answer))
        max_messages = settings.MEMORY_WINDOW * 2
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]
        cl.user_session.set("chat_history", chat_history)

    cl.user_session.set("last_question", message.content)
    cl.user_session.set("last_answer", answer)

    if not is_clarification:
        await cl.Message(
            content="Was this helpful?",
            actions=[
                cl.Action(
                    name="feedback_positive",
                    payload={"rating": "positive"},
                    label="👍  Yes",
                ),
                cl.Action(
                    name="feedback_negative",
                    payload={"rating": "negative"},
                    label="👎  No",
                ),
            ],
        ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Feedback: Positive (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("feedback_positive")
async def on_positive(action: cl.Action):
    await write_feedback(rating="positive", comment=None)
    await action.remove()
    await cl.Message(content="✅ Thanks for the feedback!").send()


# ─────────────────────────────────────────────────────────────────────────────
# Feedback: Negative (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("feedback_negative")
async def on_negative(action: cl.Action):
    await action.remove()
    cl.user_session.set("awaiting_feedback", True)
    await cl.Message(
        content="Sorry about that. What was wrong? (type below and press Enter)"
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Chat End (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_end
async def on_chat_end():
    pass