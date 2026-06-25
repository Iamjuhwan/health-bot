"""
rag_pipeline.py

This is where everything comes together:

  user message
      |
      v
  1. guardrail check  --- if triggered: return fixed safe response, STOP
      |
      v (only if not triggered)
  2. retrieve top-k relevant knowledge base entries
      |
      v
  3. build a prompt: system instructions + retrieved context + user question
      |
      v
  4. call Gemini to generate the final answer
      |
      v
  final response

The LLM call is deliberately isolated in its own function (call_llm),
so swapping Gemini for OpenAI or Claude later only means changing that
one function — nothing else in the pipeline needs to know which model
is being used.

Memory layer:
  Each Telegram chat_id gets its own conversation history (last 6 turns).
  This is passed into the prompt so the LLM can handle follow-up questions
  like "what about side effects?" without losing context.
  History lives in RAM — it resets when the bot restarts (fine for now).

Run with:
    python rag_pipeline.py "your question here"
"""

import os
import sys
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve
from guardrails import check_guardrail

load_dotenv()  # reads GEMINI_API_KEY from a local .env file, never from code

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # fast + free-tier friendly

SYSTEM_INSTRUCTIONS = """You are a friendly, supportive health and wellbeing assistant for \
adolescent girls. You answer using ONLY the context provided below — do not use outside \
knowledge, and do not guess. If the context doesn't contain a good answer, say you're not \
sure and suggest talking to a trusted adult or healthcare provider.

Keep answers short, warm, and easy to understand. Never diagnose. Never give medical, legal, \
or instructional advice beyond what's in the context. If a question is outside what you can \
safely answer, gently redirect the user to a trusted adult, counselor, or healthcare provider."""

# In-memory conversation history per Telegram chat_id
# {chat_id: [{"role": ..., "content": ...}, ...]}
conversation_history = defaultdict(list)
MAX_HISTORY = 6  # last 3 exchanges (user + assistant each)


def call_llm(user_question: str, context_chunks: list[dict], chat_id: int = 0) -> str:
    """
    Sends the user's question + retrieved context + conversation history
    to Gemini and returns the generated answer.

    chat_id is used to look up this user's conversation history so the
    model can handle follow-up questions in context.

    This is the ONLY function that knows which LLM provider we're using —
    keep it that way so swapping providers later is a one-function change.
    """
    if not GEMINI_API_KEY:
        return (
            "[No GEMINI_API_KEY found. Add it to a .env file in this folder, "
            "e.g. GEMINI_API_KEY=your_real_key]"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    context_text = "\n\n".join(
        f"- {c['question']}: {c['answer']}" for c in context_chunks
    )

    # Build conversation history block (only if there's prior context)
    history_text = ""
    if conversation_history[chat_id]:
        turns = conversation_history[chat_id][-MAX_HISTORY:]
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in turns
        )
        history_text = f"\nPrevious conversation:\n{history_text}\n"

    prompt = f"""Context (from our verified knowledge base):
{context_text}
{history_text}
User question: {user_question}

Answer the user's question using only the context above."""

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
        ),
    )

    reply = response.text.strip()

    # Save this exchange to memory
    conversation_history[chat_id].append({"role": "user", "content": user_question})
    conversation_history[chat_id].append({"role": "assistant", "content": reply})

    return reply


def answer_question(user_question: str, chat_id: int = 0, top_k: int = 3) -> dict:
    """
    The full pipeline. Returns a dict so the caller (CLI, Telegram bot,
    etc.) can see not just the final text but also whether a guardrail
    fired and what was retrieved — useful for logging and debugging.
    """
    guardrail_result = check_guardrail(user_question)

    if guardrail_result.triggered:
        return {
            "answer": guardrail_result.safe_response,
            "guardrail_triggered": True,
            "guardrail_category": guardrail_result.category,
            "retrieved_context": [],
        }

    context_chunks = retrieve(user_question, top_k=top_k)
    answer = call_llm(user_question, context_chunks, chat_id=chat_id)

    return {
        "answer": answer,
        "guardrail_triggered": False,
        "guardrail_category": None,
        "retrieved_context": context_chunks,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag_pipeline.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer_question(question)

    print(f"\nQuestion: {question}\n")
    if result["guardrail_triggered"]:
        print(f"[GUARDRAIL TRIGGERED: {result['guardrail_category']}]")
    print(f"Answer: {result['answer']}")