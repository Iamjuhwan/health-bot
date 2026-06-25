# health-bot

# 🩺 Health & Wellbeing Telegram Bot

A RAG-powered Telegram bot that answers health and wellbeing questions for young people — with guardrails, conversation memory, and voice note support.

---

## What it does

- Answers questions about puberty, periods, relationships, mental health, and personal safety
- Accepts both **text messages and voice notes**
- Transcribes voice notes via **OpenAI Whisper**, then runs them through the same pipeline as text
- Retrieves relevant context from a curated knowledge base using **FAISS vector search**
- Generates empathetic, age-appropriate responses via **Google Gemini**
- Fires **guardrails** for harmful or out-of-scope content (regex + ML classifier)
- Maintains **per-user conversation memory** across a session
- Logs guardrail triggers with timestamps for review

---

## Architecture

```
User (Telegram)
    │
    ├── Text message ─────────────────────────────┐
    │                                             ▼
    └── Voice note → Whisper transcription → Guardrail check
                                                  │
                                         ┌────────┴────────┐
                                    Triggered          Not triggered
                                         │                  │
                                   Safe reply          FAISS retrieval
                                                            │
                                                     Gemini generation
                                                            │
                                                      Reply to user
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Channel | Python Telegram Bot |
| Transcription | OpenAI Whisper (`whisper-1`) |
| Guardrails | Regex + HuggingFace classifier |
| Retrieval | FAISS + sentence-transformers |
| Generation | Google Gemini |
| Memory | In-memory per `chat_id` |

---

## Project structure

```
health-bot/
├── telegram_bot.py       # Channel layer — routes text and voice messages
├── rag_pipeline.py       # Guardrail → retrieve → generate + memory
├── guardrails.py         # Regex patterns + ML classifier
├── retrieve.py           # FAISS vector search
├── voice.py              # Whisper transcription
├── data/
│   └── qa_pairs.json     # Curated knowledge base
└── .env                  # API keys (not committed)
```

---

## Running locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/health-bot.git
cd health-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

- Get a Telegram bot token from [@BotFather](https://t.me/botfather)
- Get a Gemini key from [Google AI Studio](https://aistudio.google.com)
- Get an OpenAI key from [platform.openai.com](https://platform.openai.com)

### 4. Run the bot

```bash
python telegram_bot.py
```

---

## Guardrail behaviour

The bot checks every message (text or transcribed voice) before passing it to the RAG pipeline.

| Input | Behaviour |
|---|---|
| Health question | Retrieves context, generates answer |
| Self-harm mention | Guardrail fires, safe signposting response |
| Out-of-scope topic | Guardrail fires, redirect message |
| Silent/corrupted audio | Error caught, fallback message shown |

Guardrail triggers are logged with user ID, category, and timestamp.

---

## Deploying to Railway

See [DEPLOY.md](./DEPLOY.md) for step-by-step deployment instructions.

---

## Limitations & future work

- Conversation memory resets when the server restarts (no persistent storage yet)
- Voice transcription is English-only by default (removable in `voice.py`)
- Knowledge base is static — a future version could allow admin updates via a simple web UI

---

## Disclaimer

This bot is not a substitute for professional medical advice, a doctor, or a counsellor. It is designed to provide general information and signpost users to real help where appropriate.
