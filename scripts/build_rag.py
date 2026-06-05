"""
Builds the mental-health RAG store from docs/*.md, docs/*.txt, and docs/*.pdf.

Two embedding backends, picked at build time:
  - 'sbert' -> sentence-transformers/all-MiniLM-L6-v2 (semantic, ~80 MB download)
  - 'tfidf' -> scikit-learn TF-IDF (fully offline, no LLM/HF deps)

Default tries sbert and falls back to tfidf if the model can't be loaded.
"""
import json
import os
import pickle

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(ROOT, "docs")
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


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Install pypdf to ingest PDF references: pip install pypdf") from e

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_reference_docs():
    docs = []
    for name in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in {".md", ".txt"}:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            text = read_pdf(path)
        else:
            continue
        if text.strip():
            docs.append((name, text))
    if not docs:
        raise RuntimeError(f"No reference docs found in {DOCS_DIR}")
    return docs


def embed_sbert(chunks):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = m.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32), "sbert"


def embed_tfidf(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(chunks).toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X = X / norms
    with open(os.path.join(STORE_DIR, "tfidf_vec.pkl"), "wb") as f:
        pickle.dump(vec, f)
    return X, "tfidf"


def main():
    import faiss

    docs = load_reference_docs()
    chunks = []
    doc_names = []
    for name, raw in docs:
        doc_chunks = chunk_text(raw)
        chunks.extend(doc_chunks)
        doc_names.append({"name": name, "chunks": len(doc_chunks)})
    print(f"[rag] {len(chunks)} chunks from {len(doc_names)} document(s)")

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
    with open(os.path.join(STORE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": backend, "dim": int(embs.shape[1]), "documents": doc_names}, f)
    print(f"[rag] saved FAISS index -> {STORE_DIR}")


if __name__ == "__main__":
    main()
