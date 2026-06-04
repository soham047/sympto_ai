"""
RAG-based mental health chatbot.

Pipeline: query -> embed (SBERT or TF-IDF fallback) -> FAISS top-k -> grounded
answer (flan-t5-small if available, else extractive top-chunk answer).
"""
import os
import json
import pickle
import numpy as np
from flask import Blueprint, request, jsonify

bp = Blueprint("chat", __name__)

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "faiss_store")

_state = {"index": None, "chunks": None, "backend": None, "embedder": None, "tfidf": None,
          "llm": None, "tokenizer": None, "llm_tried": False}


def _load_rag():
    if _state["index"] is not None:
        return
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("faiss-cpu not installed: pip install faiss-cpu") from e

    idx_path = os.path.join(STORE_DIR, "index.faiss")
    chk_path = os.path.join(STORE_DIR, "chunks.json")
    meta_path = os.path.join(STORE_DIR, "meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(chk_path) and os.path.exists(meta_path)):
        raise RuntimeError("FAISS store not built — run: python scripts/build_rag.py")

    with open(meta_path) as f:
        meta = json.load(f)
    backend = meta["backend"]

    if backend == "sbert":
        from sentence_transformers import SentenceTransformer
        _state["embedder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    elif backend == "tfidf":
        with open(os.path.join(STORE_DIR, "tfidf_vec.pkl"), "rb") as f:
            _state["tfidf"] = pickle.load(f)
    else:
        raise RuntimeError(f"unknown backend {backend}")

    _state["index"] = faiss.read_index(idx_path)
    with open(chk_path, "r", encoding="utf-8") as f:
        _state["chunks"] = json.load(f)
    _state["backend"] = backend


def _embed_query(q):
    if _state["backend"] == "sbert":
        return _state["embedder"].encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    X = _state["tfidf"].transform([q]).toarray().astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n


def _try_load_llm():
    if _state["llm"] is not None or _state["llm_tried"]:
        return _state["llm"] is not None
    _state["llm_tried"] = True
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        _state["tokenizer"] = T5Tokenizer.from_pretrained("google/flan-t5-small")
        _state["llm"] = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        return True
    except Exception as e:
        print(f"[chat] LLM unavailable, using extractive fallback ({type(e).__name__})")
        return False


def retrieve(query, k=3):
    _load_rag()
    q_emb = _embed_query(query)
    scores, ids = _state["index"].search(q_emb, k)
    out = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        out.append({"text": _state["chunks"][i], "score": float(s)})
    return out


SAFETY_NOTE = (
    "\n\n— Educational info only, not a substitute for professional care. "
    "If you are in crisis, contact your local emergency line or a crisis helpline."
)

CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "self harm", "self-harm", "hurt myself"]


def crisis_check(query):
    q = query.lower()
    if any(kw in q for kw in CRISIS_KEYWORDS):
        return ("It sounds like you're going through something really hard right now. "
                "Please reach out to a crisis line immediately — US: 988, UK Samaritans: 116 123, "
                "India iCall: 9152987821, Australia Lifeline: 13 11 14. "
                "You can also go to the nearest emergency department. You don't have to face this alone.")
    return None


@bp.route("", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "empty message"}), 400

    c = crisis_check(query)
    if c:
        return jsonify({"answer": c, "sources": []})

    try:
        hits = retrieve(query, k=3)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    context = "\n".join(h["text"] for h in hits)

    if _try_load_llm():
        prompt = (
            "You are a supportive mental-health information assistant. Using only the context below, "
            "answer the user's question in 2-4 sentences. If the context does not contain the answer, "
            "say you don't have that information and suggest seeing a mental-health professional.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        inputs = _state["tokenizer"](prompt, return_tensors="pt", truncation=True, max_length=1024)
        out = _state["llm"].generate(**inputs, max_new_tokens=180, num_beams=2)
        answer = _state["tokenizer"].decode(out[0], skip_special_tokens=True)
    else:
        answer = hits[0]["text"] if hits else "I don't have information on that. Please consider speaking with a professional."

    return jsonify({
        "answer": answer + SAFETY_NOTE,
        "sources": [{"text": h["text"][:200] + ("…" if len(h["text"]) > 200 else ""),
                     "score": round(h["score"], 3)} for h in hits],
        "backend": _state["backend"],
    })
