"""
retrieve.py

Step 2 of the RAG pipeline: given a user's question, find the most
relevant entries from the knowledge base.

This is "retrieval" — the R in RAG. It does NOT call any LLM. It just
finds the best-matching pieces of our knowledge base using vector
similarity search.

Usage (from the command line, for quick testing):
    python retrieve.py "what should I do if my period is really painful"
"""

import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/kb_index.faiss"
METADATA_PATH = "data/kb_metadata.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once and reused — re-loading the model on every query would be slow.
_model = None
_index = None
_metadata = None


def _load_resources():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _metadata is None:
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Returns the top_k most relevant knowledge base entries for the query,
    each with a similarity score between 0 and 1 (higher = more relevant).
    """
    _load_resources()

    query_vector = _model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = _index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = _metadata[idx]
        results.append({
            "score": float(score),
            "id": entry["id"],
            "topic": entry["topic"],
            "question": entry["question"],
            "answer": entry["answer"],
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrieve.py \"your question here\"")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    matches = retrieve(user_query, top_k=3)

    print(f"\nQuery: {user_query}\n")
    print("Top matches:")
    for i, m in enumerate(matches, 1):
        print(f"\n{i}. [score={m['score']:.3f}] (topic: {m['topic']})")
        print(f"   Q: {m['question']}")
        print(f"   A: {m['answer']}")
