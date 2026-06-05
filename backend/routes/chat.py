"""Groq-powered mental health chatbot with quiet RAG grounding."""
import json
import os
import pickle
import re
from urllib import error, request as urlrequest

import numpy as np
from flask import Blueprint, jsonify, request

bp = Blueprint("chat", __name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORE_DIR = os.path.join(ROOT, "models", "faiss_store")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_state = {"index": None, "chunks": None, "backend": None, "embedder": None, "tfidf": None}


def _load_local_env():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
        raise RuntimeError("FAISS store not built - run: python scripts/build_rag.py")

    with open(meta_path, encoding="utf-8") as f:
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


CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "want to die",
    "dont want to live",
    "don't want to live",
]
OFF_TOPIC_HINTS = [
    "calculate",
    "solve",
    "equation",
    "derivative",
    "integral",
    "homework",
    "code this",
    "write code",
    "stock price",
    "weather",
    "recipe",
]


def crisis_check(query):
    q = query.lower()
    if any(kw in q for kw in CRISIS_KEYWORDS):
        return (
            "I'm really glad you said something. If you might hurt yourself or you feel in immediate danger, "
            "please call emergency services now or contact a crisis line: US 988, UK Samaritans 116 123, "
            "India iCall 9152987821, Australia Lifeline 13 11 14. If you can, move near another person and "
            "tell them you need help right now."
        )
    return None


def _looks_off_topic(query):
    q = query.lower()
    return any(hint in q for hint in OFF_TOPIC_HINTS)


def _clean_answer(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"(?i)\b(source|reference|file|line|citation)s?\s*[:#-]?\s*\S*", "", text).strip()
    text = re.sub(r"\[(?:\d+|source[^\]]*)\]", "", text, flags=re.I).strip()
    return text


def _groq_chat(query, context):
    _load_local_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    if _looks_off_topic(query):
        system = (
            "You are a friendly mental-health and wellness chatbot. The user is asking outside your scope. "
            "Reply naturally in 1-2 short sentences and gently say you are best for mental-health, stress, "
            "sleep, emotions, coping, and wellbeing questions. Do not solve math, coding, finance, weather, "
            "or unrelated tasks."
        )
        user = query
    else:
        system = (
            "You are a warm, normal, fast mental-health support chatbot. Use the provided reference context "
            "quietly when it helps, but never mention sources, files, references, PDFs, line numbers, retrieval, "
            "or citations. Sound like a natural chatbot, not a report. Keep replies short unless the user asks "
            "for detail. Do not diagnose. For casual greetings or ordinary small talk, respond casually. For "
            "mental-health concerns, be supportive and practical. For crisis or self-harm content, urge immediate "
            "human help and crisis support."
        )
        user = f"Reference context:\n{context}\n\nUser message: {query}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.45,
        "max_tokens": 260,
    }
    req = urlrequest.Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sympto-ai/1.0",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"Groq request failed ({e.code}): {detail}") from e
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {type(e).__name__}") from e
    return _clean_answer(data["choices"][0]["message"]["content"])


def _fallback_answer(query, hits):
    if _looks_off_topic(query):
        return "I'm best at mental-health, stress, sleep, emotions, coping, and wellbeing questions. Ask me something in that space and I'll help."

    q = query.lower()
    if any(word in q for word in ["anxious", "anxiety", "panic", "calm"]):
        return (
            "That sounds really uncomfortable. Try slowing things down for the next minute: breathe out longer "
            "than you breathe in, unclench your shoulders, and name five things you can see. Then tell me what "
            "set the anxiety off, if you want to talk it through."
        )
    if any(word in q for word in ["sleep", "insomnia", "tired"]):
        return (
            "Sleep trouble can feel miserable. Tonight, keep it simple: dim the lights, put the phone away for a bit, "
            "and do something boring and quiet until your body starts to settle. If this has been going on for weeks, "
            "it may be worth talking with a clinician."
        )
    if any(word in q for word in ["burnout", "overwhelmed", "stress", "stressed"]):
        return (
            "That sounds like a lot to carry. Start by lowering the load for the next hour, not solving everything: "
            "drink water, step away for a few minutes, and pick one tiny task that would make the day feel less sharp."
        )
    if any(word in q for word in ["sad", "depressed", "low", "lonely"]):
        return (
            "I'm sorry you're feeling this way. You don't have to make the whole feeling disappear right now; just try "
            "one grounding thing, like eating something small, messaging someone safe, or sitting somewhere with light."
        )
    return "I'm here with you. Tell me a little more about what's been going on, and I'll try to help you sort through it."


@bp.route("", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "empty message"}), 400

    c = crisis_check(query)
    if c:
        return jsonify({"answer": c, "sources": [], "model": "safety"})

    try:
        hits = retrieve(query, k=3)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    context = "\n\n".join(h["text"] for h in hits)
    try:
        answer = _groq_chat(query, context)
    except RuntimeError as e:
        print(f"[chat] {e}; using local fallback")
        answer = _fallback_answer(query, hits)

    return jsonify({"answer": answer, "sources": [], "backend": _state["backend"], "model": GROQ_MODEL})
