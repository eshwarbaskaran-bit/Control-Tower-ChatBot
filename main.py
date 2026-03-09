import sys
import asyncio

# THE WINDOWS ASYNC FIX: Force Windows to use the correct Event Loop for Postgres
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from agent import SemanticSniperAgent

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="🏗️ Booting ClickPost Logic Engine...")
    await msg.send()
    
    try:
        # 1. Instantiate the bot WITH the async flag set to True
        bot = SemanticSniperAgent(is_async=True)
        
        # 2. Save it to the session
        cl.user_session.set("bot", bot)
        
        msg.content = "✅ **Control Tower Online.** Ask me about logistics logic."
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ **Startup Error:** {str(e)}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    # 3. Retrieve the bot
    bot = cl.user_session.get("bot")
    
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    try:
        # 4. Use the .aask() method for Chainlit
        answer = await bot.aask(message.content)
        msg.content = answer
    except Exception as e:
        msg.content = f"⚠️ Processing Error: {str(e)}"
        
    await msg.update()