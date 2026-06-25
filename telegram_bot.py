"""
telegram_bot.py

Deploys the RAG pipeline as a live Telegram bot.

This is the "channel" layer — everything before this (retrieval,
guardrails, generation) doesn't know or care that it's being used via
Telegram. That separation is intentional: the same rag_pipeline.py
could be wired up to WhatsApp, a web widget, or any other channel
without touching the pipeline logic itself.

Supports:
- Text messages → RAG pipeline
- Voice messages → Whisper transcription → RAG pipeline
- /start command → welcome message
- /clear command → reset conversation memory

Run with:
    python telegram_bot.py
"""

import os
import logging
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

from rag_pipeline import answer_question, conversation_history
from voice import transcribe_voice

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "Hi! I'm a health and wellbeing helper. You can ask me questions about "
        "puberty, periods, relationships, safety, or how you're feeling — I'll "
        "do my best to give you clear, supportive answers.\n\n"
        "I'm not a substitute for a doctor, counselor, or trusted adult, and if "
        "something serious is going on, I'll always point you toward real help.\n\n"
        "You can type your question or send a voice note 🎙️\n\n"
        "What would you like to ask?"
    )
    await update.message.reply_text(welcome_message)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    conversation_history[chat_id].clear()
    await update.message.reply_text(
        "Got it — I've forgotten our previous conversation. Fresh start! 🔄"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id

    logger.info(f"Text message from user {user_id}: {user_message}")

    result = answer_question(user_message, chat_id=chat_id)

    if result["guardrail_triggered"]:
        logger.warning(
            f"GUARDRAIL TRIGGERED for user {user_id} "
            f"(category: {result['guardrail_category']}) at {datetime.now().isoformat()}"
        )

    await update.message.reply_text(result["answer"])


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id

    logger.info(f"Voice message received from user {user_id}")

    # Let the user know we're processing
    await update.message.reply_text("🎙️ Got your voice note, transcribing...")

    try:
        # Download and transcribe
        file = await context.bot.get_file(update.message.voice.file_id)
        transcribed_text = await transcribe_voice(file)

        # Show user what was transcribed so they can confirm it was heard correctly
        await update.message.reply_text(f"📝 I heard: _{transcribed_text}_", parse_mode="Markdown")

        # Run through the exact same pipeline as a text message
        result = answer_question(transcribed_text, chat_id=chat_id)

        if result["guardrail_triggered"]:
            logger.warning(
                f"GUARDRAIL TRIGGERED (voice) for user {user_id} "
                f"(category: {result['guardrail_category']}) at {datetime.now().isoformat()}"
            )

        await update.message.reply_text(result["answer"])

    except Exception as e:
        logger.error(f"Voice handling error for user {user_id}: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't process that voice note. Could you try typing your question instead?"
        )


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: No TELEGRAM_BOT_TOKEN found. Add it to your .env file.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))  # voice handler

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()