"""
build_index.py

Step 1 of the RAG pipeline: turn the knowledge base into a searchable vector index.

What this does:
1. Loads data/knowledge_base.json (our Q&A pairs)
2. Converts each entry into a numeric vector ("embedding") using a free,
   local model (no API key needed for this step)
3. Stores those vectors in a FAISS index so we can quickly find the
   most relevant entries for any incoming question
4. Saves the index + the original text to disk, so retrieve.py can load
   it later without re-computing embeddings every time

Run this once whenever the knowledge base changes:
    python build_index.py
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

KB_PATH = "data/knowledge_base.json"
INDEX_PATH = "data/kb_index.faiss"
METADATA_PATH = "data/kb_metadata.pkl"

# all-MiniLM-L6-v2 is a small, fast, free embedding model — good enough
# for a knowledge base this size, and light enough to run on a laptop.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    print(f"Loading knowledge base from {KB_PATH} ...")
    kb = load_knowledge_base(KB_PATH)
    print(f"Loaded {len(kb)} Q&A entries.")

    # We embed "question + answer" together so retrieval matches both
    # how a user might phrase a question AND the content of the answer.
    texts_to_embed = [f"{item['question']} {item['answer']}" for item in kb]

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Computing embeddings (this may take a minute the first time) ...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors so we can use inner product as cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine sim after normalization)
    index.add(embeddings)

    print(f"Built FAISS index with {index.ntotal} vectors of dimension {dimension}.")

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(kb, f)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print("Done. You can now run retrieve.py to test retrieval.")


if __name__ == "__main__":
    build_index()
