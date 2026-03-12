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
from agent_render import SemanticSniperAgent


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
# Supabase feedback writer
# ─────────────────────────────────────────────────────────────────────────────
async def write_feedback(rating: str, comment: str | None):
    raw_url = os.environ.get("SUPABASE_FEEDBACK_URL", "")
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
# Chat Start
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="🏗️ Booting ClickPost Logic Engine...")
    await msg.send()

    try:
        bot = SemanticSniperAgent(is_async=True)
        cl.user_session.set("bot", bot)
        cl.user_session.set("last_question", "")
        cl.user_session.set("last_answer", "")
        cl.user_session.set("awaiting_feedback", False)

        user = cl.context.session.user
        cl.user_session.set("username", user.identifier if user else "clickpost")

        msg.content = "✅ **Control Tower Online.** Ask me anything about ClickPost logistics."
        await msg.update()

    except Exception as e:
        msg.content = f"❌ **Startup Error:** {str(e)}"
        await msg.update()


# ─────────────────────────────────────────────────────────────────────────────
# On Message
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):

    # ── Intercept free-text feedback ─────────────────────────────────────────
    if cl.user_session.get("awaiting_feedback", False):
        cl.user_session.set("awaiting_feedback", False)
        await write_feedback(rating="negative", comment=message.content)
        await cl.Message(content="📝 Got it — feedback saved. Thanks!").send()
        return

    # ── Normal bot response ───────────────────────────────────────────────────
    bot: SemanticSniperAgent = cl.user_session.get("bot")

    msg = cl.Message(content="")
    await msg.send()

    try:
        response = await bot.retrieval_chain.ainvoke(
            {"input": message.content},
            config={
                "tags": ["control-tower", "chainlit"],
                "metadata": {
                    "session_id": cl.context.session.id,
                    "user_query": message.content,
                },
            },
        )
        answer = response["answer"]

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    msg.content = answer
    await msg.update()

    cl.user_session.set("last_question", message.content)
    cl.user_session.set("last_answer", answer)

    # ── Feedback buttons ──────────────────────────────────────────────────────
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
# Feedback: Positive
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("feedback_positive")
async def on_positive(action: cl.Action):
    await write_feedback(rating="positive", comment=None)
    await action.remove()
    await cl.Message(content="✅ Thanks for the feedback!").send()


# ─────────────────────────────────────────────────────────────────────────────
# Feedback: Negative
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("feedback_negative")
async def on_negative(action: cl.Action):
    await action.remove()
    cl.user_session.set("awaiting_feedback", True)
    await cl.Message(
        content="Sorry about that. What was wrong? (type below and press Enter)"
    ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Chat End
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_end
async def on_chat_end():
    pass