"""
Builds the mental-health RAG store.

Two embedding backends, picked at build time:
  - 'sbert'  → sentence-transformers/all-MiniLM-L6-v2 (semantic, ~80 MB download)
  - 'tfidf'  → scikit-learn TF-IDF (fully offline, no LLM/HF deps)

Default tries sbert and falls back to tfidf if the model can't be loaded.
"""
import os
import json
import pickle
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC_PATH = os.path.join(ROOT, "docs", "mental_health_reference.md")
STORE_DIR = os.path.join(ROOT, "models", "faiss_store")
os.makedirs(STORE_DIR, exist_ok=True)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_sbert(chunks):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = m.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32), "sbert"


def embed_tfidf(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(chunks).toarray().astype(np.float32)
    # L2-normalize so inner product == cosine similarity (matches FAISS IndexFlatIP semantics)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms
    with open(os.path.join(STORE_DIR, "tfidf_vec.pkl"), "wb") as f:
        pickle.dump(vec, f)
    return X, "tfidf"


def main():
    import faiss

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = chunk_text(raw)
    print(f"[rag] {len(chunks)} chunks")

    try:
        embs, backend = embed_sbert(chunks)
        print(f"[rag] using SBERT embeddings (dim={embs.shape[1]})")
    except Exception as e:
        print(f"[rag] SBERT unavailable ({type(e).__name__}: {str(e)[:80]}...). Falling back to TF-IDF.")
        embs, backend = embed_tfidf(chunks)
        print(f"[rag] using TF-IDF embeddings (dim={embs.shape[1]})")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(STORE_DIR, "index.faiss"))
    with open(os.path.join(STORE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(os.path.join(STORE_DIR, "meta.json"), "w") as f:
        json.dump({"backend": backend, "dim": int(embs.shape[1])}, f)
    print(f"[rag] saved FAISS index -> {STORE_DIR}")


if __name__ == "__main__":
    main()
