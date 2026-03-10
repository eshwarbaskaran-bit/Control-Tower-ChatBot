import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from agent import SemanticSniperAgent

# ─────────────────────────────────────────────────────────────────────────────
# Data Layer — persists chat history AND feedback (thumbs up/down) to SQLite.
# ─────────────────────────────────────────────────────────────────────────────
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///chainlit_history.db")

# ─────────────────────────────────────────────────────────────────────────────
# Authentication — REQUIRED for Chainlit to display history in the UI
# ─────────────────────────────────────────────────────────────────────────────
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    # Dummy login for local testing
    if username == "eshwar" and password == "admin":
        return cl.User(identifier="eshwar")
    return None

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
        msg.content = "✅ **Control Tower Online.** Ask me anything about ClickPost logistics."
        await msg.update()
    except Exception as e:
        msg.content = f"❌ **Startup Error:** {str(e)}"
        await msg.update()

# ─────────────────────────────────────────────────────────────────────────────
# On Message — with LangSmith run_id surfaced for traceability
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    bot: SemanticSniperAgent = cl.user_session.get("bot")

    msg = cl.Message(content="")
    await msg.send()

    try:
        # ainvoke directly so we can capture run metadata for LangSmith
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
        msg.content = answer

    except Exception as e:
        msg.content = f"⚠️ Error: {str(e)}"

    await msg.update()

@cl.on_chat_end
async def on_chat_end():
    pass