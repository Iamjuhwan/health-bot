"""
voice.py

Handles voice message transcription for the Telegram bot.

Flow:
    Telegram voice note → download → Whisper API → transcribed text

The transcribed text is then passed into the same answer_question()
pipeline as any text message — the pipeline doesn't know or care
whether the input came from typing or speaking.
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def transcribe_voice(file) -> str:
    """
    Downloads a Telegram voice file and transcribes it using Whisper.

    `file` is a telegram.File object — call this after doing:
        file = await context.bot.get_file(update.message.voice.file_id)

    Returns the transcribed text string, or raises an exception if
    transcription fails.
    """
    # Download the voice note as bytes
    voice_bytes = await file.download_as_bytearray()

    # Whisper expects a file-like object with a name so it knows the format
    import io
    audio_file = io.BytesIO(bytes(voice_bytes))
    audio_file.name = "voice.ogg"  # Telegram sends voice notes as .ogg

    logger.info("Sending voice note to Whisper for transcription...")

    transcript = _openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",  # remove this line if you want auto language detection
    )

    transcribed_text = transcript.text.strip()
    logger.info(f"Transcribed: {transcribed_text}")

    return transcribed_text