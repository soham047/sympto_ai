"""Run every trainer + build the RAG index. ~1-2 minutes on CPU.

Gracefully skips steps whose dependencies aren't installed, so you can still
get a partial app running with only scikit-learn available.
"""
import importlib
import sys
import os

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

STEPS = [
    ("train_blood",     []),                              # sklearn only
    ("train_symptom",   []),
    ("train_lifestyle", []),
    ("train_xray",      ["tensorflow"]),
    ("train_skin",      ["tensorflow"]),
    ("build_rag",       ["faiss"]),                       # falls back to TF-IDF if sbert unavailable
]


def have(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


skipped = []
for name, deps in STEPS:
    missing = [d for d in deps if not have(d)]
    if missing:
        print(f"\n--- skipping {name}: missing {missing}")
        skipped.append((name, missing))
        continue
    print(f"\n=== running {name} ===")
    mod = importlib.import_module(name)
    mod.main()

print("\nDone.")
if skipped:
    print("\nSkipped steps (install their deps to enable):")
    for n, m in skipped:
        print(f"  {n} -> pip install {' '.join(m)}")
print("\nStart the app:  python backend/app.py")
