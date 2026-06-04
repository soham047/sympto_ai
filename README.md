# AI-Driven Healthcare Platform

A student-level integrated platform with 5 healthcare AI features behind a single Flask backend + basic HTML frontend.

## Features

| # | Feature | Model | Tech |
|---|---------|-------|------|
| 1 | **Blood Report Analyzer** | Tuned CBC classifier from `data/blood_report_dataset.xlsx` | scikit-learn |
| 2 | **Symptom Checker** | Tuned symptom classifier from `data/symptom_checker_dataset.xlsx` | scikit-learn |
| 3 | **Image Diagnosis** | CNN from scratch + MobileNetV2 fine-tune (X-ray & Skin) | TensorFlow/Keras |
| 4 | **Mental Health Chatbot** | RAG (FAISS + sentence-transformers + flan-t5) | LangChain-style pipeline |
| 5 | **Lifestyle Planner** | Tuned calorie/protein regression + planning engine | scikit-learn |

## Project structure

```
ai-healthcare-platform/
├── notebooks/         # Full EDA + training notebooks
├── scripts/           # train_*.py training entry points
├── backend/           # Flask app with /blood /symptom /image /chat /lifestyle
├── frontend/          # 5-page basic HTML/CSS/JS UI
├── models/            # Pickled / h5 / FAISS artifacts (created by training scripts)
├── data/              # Datasets (gitignored — download separately)
└── docs/              # Mental-health reference doc used by the RAG chatbot
```

## Quick start

> **Use Python 3.11 or 3.12** for the full stack. TensorFlow currently has no
> wheels for Python 3.13/3.14, so the X-ray and Skin CNN routes won't run on
> those versions (the other 3 features still work).

```powershell
# 1. install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. train all available models from local data/artifacts
python scripts/train_all.py

# 3. run backend
python backend/app.py

# 4. open frontend
# visit http://127.0.0.1:5000  in your browser
```

### Partial install (sklearn-only)
If you only have scikit-learn + Flask installed, `train_all.py` will train the 3 tabular models and skip the CNN/RAG steps gracefully. The corresponding API endpoints return a clear `503` with instructions when called.

### RAG backend modes
`scripts/build_rag.py` tries `sentence-transformers/all-MiniLM-L6-v2` (semantic) and falls back to TF-IDF embeddings if that fails or isn't installed. Either way it produces a FAISS index the chat route reads. For best quality, install sentence-transformers and re-run the script.

## Datasets

The blood and symptom datasets are already expected at:

- `data/blood_report_dataset.xlsx`
- `data/symptom_checker_dataset.xlsx`

Their notebooks and scripts clean those Excel files, run EDA/visualisations, compare multiple ML and neural-network-style trials, tune models, and save the corresponding artifacts in `models/`.

Additional optional datasets for the image/lifestyle modules:

| Feature | Dataset | Link |
|---------|---------|------|
| X-ray | Pneumonia Chest X-Ray | Kaggle: `paultimothymooney/chest-xray-pneumonia` |
| Skin | HAM10000 | Kaggle: `kmader/skin-cancer-mnist-ham10000` |
| Mental Health RAG | NIMH/WHO fact sheets | Free PDFs at nimh.nih.gov / who.int |
| Lifestyle | Optional fitness / calories references | Kaggle: `fmendes/fmendesdat263xdemos` |

Place downloaded optional data inside `data/<feature>/` and rerun the notebook for that feature.

## Notebooks

Each notebook does **proper EDA → visualisations → multiple model implementations → DL fine-tuning** so it doubles as a portfolio artifact:

- `notebooks/01_blood_report_analyzer.ipynb`
- `notebooks/02_symptom_checker.ipynb`
- `notebooks/03_xray_cnn.ipynb`
- `notebooks/04_skin_disease_cnn.ipynb`
- `notebooks/05_mental_health_rag.ipynb`
- `notebooks/06_lifestyle_planner.ipynb`

## Disclaimer

This is a learning project — **not** a medical device. Predictions are illustrative only.
