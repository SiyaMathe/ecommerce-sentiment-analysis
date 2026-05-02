# 🛒 Amazon Appliance Review Sentiment Analysis
## LSTM vs Conv1D vs Bidirectional LSTM — NLP Binary Classification with GloVe Embeddings

**Author:** Siyabulela Mathe  
**Dataset:** Amazon Appliance Reviews (`Appliances.json`)  
**Stack:** PySpark · TensorFlow/Keras · NLTK · GloVe Embeddings  
**Task:** Binary sentiment classification — Positive (> 3★) vs Negative (≤ 3★)

---

## What This Project Does

Given a customer product review such as:

> *"Purchased this for my kitchen. Initially worked great but stopped working after 3 weeks. Very disappointed."*

The model predicts: **Negative** (despite the positive opening — the BiLSTM reads both directions and catches the reversal).

---

## Key Fixes Over the Original Baseline Code

| Bug | Original (broken) | Fixed |
|---|---|---|
| PySpark version | `pyspark.version` → `AttributeError` | `pyspark.__version__` |
| PySpark loading | `SQLContext.read.json()` on the **class** → `AttributeError` | `spark.read.json()` on the **instance** |
| JSON format | `multiLine=True` unconditionally → corrupt records on most LC files | Auto-detects JSON-lines vs multiLine |
| Pandas on Spark DF | `.isnull()`, `.shape` on Spark DF → `AttributeError` | `F.count(F.when(...))`, `.count()`, `len(.columns)` |
| Tokeniser leakage | `fit_on_texts()` before split → test vocab leaks into train | Split first, then fit on **train only** |
| Keras metric key | `history['acc']` → `KeyError` in TF2 | `history['accuracy']` (correct TF2 key) |
| No callbacks | Training ran fixed epochs — no early stopping | EarlyStopping + ModelCheckpoint + ReduceLROnPlateau |
| Single model | Only LSTM trained; Conv1D imported but never used | LSTM + Conv1D + BiLSTM — all three compared |
| Evaluation | Accuracy only | Accuracy + Precision + Recall + F1 + ROC-AUC + PR-AUC |

---

## Dataset — Amazon Appliance Reviews

| Property | Details |
|---|---|
| **Source** | Amazon product reviews (UCSD Julian McAuley dataset) |
| **File** | `Appliances.json` (JSON-lines format) |
| **Columns used** | `reviewText`, `overall` (star rating 1–5) |
| **Labelling** | Rating > 3.0 → Positive (1) \| Rating ≤ 3.0 → Negative (0) |
| **Download** | https://nijianmo.github.io/amazon/index.html |

---

## GloVe Embeddings

| Property | Details |
|---|---|
| **Source** | Stanford NLP Group — Global Vectors for Word Representation |
| **File** | `glove.6B.100d.txt` (100-dimensional, 6B token corpus) |
| **Download** | https://nlp.stanford.edu/projects/glove/ |
| **Vocabulary** | 400,000 English words |
| **Why GloVe?** | Pre-trained semantic structure: `king - man + woman ≈ queen`. Captures sentiment-relevant relationships like `good ↔ great ↔ excellent` and `terrible ↔ awful ↔ broken` |

Place the GloVe file in the project root:
```
sentiment-analysis/
└── a2_glove.6B.100d.txt   ← rename to this or update GLOVE_PATH in notebook
```

---

## Architecture Comparison

| Model | How it works | Best for |
|---|---|---|
| **LSTM** | Reads review left-to-right, maintains cell state across all 100 tokens | Long-range dependencies: "Great product... but stopped working" |
| **Conv1D** | Slides a 5-word filter window, GlobalMaxPooling picks the strongest signal | Local n-gram patterns: "not good", "highly recommend", "waste of money" |
| **BiLSTM** | Reads forward AND backward, concatenates both hidden states | Reviews that reverse sentiment mid-way through |

---

## Project Structure

```
sentiment-analysis/
├── appliance_sentiment_analysis.ipynb   ← Main notebook (run this)
├── README.md
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                       ← GitHub Actions (5 checks)
├── src/
│   ├── models/
│   │   └── sentiment_models.py          ← LSTM, Conv1D, BiLSTM builders + callbacks
│   ├── data/
│   │   └── preprocessing.py             ← PySpark loading, text cleaning, tokenisation, GloVe
│   ├── evaluation/
│   │   ├── metrics.py                   ← accuracy, precision, recall, F1, ROC-AUC, PR-AUC
│   │   └── business_impact.py           ← ZAR cost-benefit analysis + threshold sensitivity
│   └── visualisation/
│       └── plots.py                     ← all charts and heatmaps
├── tests/
│   └── test_pipeline.py                 ← 40+ unit tests (CS + DE + DS)
├── data/
│   └── Appliances.json                  ← Download separately
├── models/                              ← Saved .keras weights (generated)
└── reports/
    └── figures/                         ← Generated charts (saved during notebook run)
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/SiyaMathe/sentiment-analysis-appliances.git
cd sentiment-analysis-appliances

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# Run tests
pytest tests/ -v

# Launch notebook
jupyter lab appliance_sentiment_analysis.ipynb
```

---

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Environment setup — imports, config, seeds, directory creation |
| 2 | PySpark session + data loading — fixed `spark.read.json()` |
| 3 | PySpark EDA — `groupBy().count()`, rating distribution, review length |
| 4 | Text preprocessing — HTML removal, punctuation, stopwords, cleaned samples |
| 5 | Train/test split → Tokenisation (train only) → GloVe embedding matrix |
| 6 | Shared utilities — callbacks, `evaluate_model()`, `build_embedding_layer()` |
| 7 | **Model 1 — LSTM** (128 units, GloVe frozen) |
| 8 | **Model 2 — Conv1D** (128 filters, kernel=5, GlobalMaxPool) |
| 9 | **Model 3 — Bidirectional LSTM** (64 units × 2 directions) |
| 10 | Training history — loss + accuracy curves for all three models |
| 11 | Evaluation — confusion matrices, ROC curves, PR curves, scorecard |
| 12 | **Business impact** — ZAR waterfall, DL vs baseline, annual projection, threshold sensitivity |
| 13 | Summary — all fixes and key findings |

---
---

## ── Key Engineering Improvements

The following technical debt and bugs from the initial baseline were resolved to ensure production stability:

| Category | Bug Fixed | Impact |
| :--- | :--- | :--- |
| **PySpark** | `pyspark.__version__` & `spark.read.json()` | Fixed session instantiation and version attribute errors. |
| **Data Engineering** | Distributed `F.when()` scaling | Replaced slow Python loops with native Spark transforms for labelling. |
| **NLP Pipeline** | Tokenizer fit on **Train Only** | Eliminated vocabulary leakage from the test set. |
| **Deep Learning** | Deprecated Keras keys & EarlyStopping | Updated to TF2 standards and added overtraining protection. |
| **Evaluation** | Full Metric Suite | Added F1, ROC-AUC, and PR-AUC for imbalanced data. |
| **Business** | ZAR Impact Module | Added financial waterfall and threshold optimization. |

---

### ── Technical Scorecard (Final Results)

Evaluation based on a vocabulary of **62,267 words** using frozen **GloVe 100d** embeddings:

| Model | Accuracy | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| LSTM | 0.9189 | 0.9509 | 0.9515 | 0.9391 |
| Conv1D | 0.9058 | 0.9439 | 0.9386 | 0.9176 |
| **BiLSTM** 🏆 | **0.9194** | **0.9517** | **0.9537** | **0.9315** |

---

## ── Business Impact Summary (ZAR)

The model's performance was translated into a South African business context (ZAR):

*   **Deep Learning Advantage**: **R181,102,210** (Total value generated over keyword baseline on test set).
*   **Annual Net Value**: **R279,613,725** (Projected for 500k reviews/year).
*   **Annual Churn Saved**: **R7,534,642** (Estimated at 8% probability per missed negative).
*   **Strategic Optimization**: Optimal decision threshold set at **0.060**, prioritizing a **99.9% Recall** to avoid the high cost of missed negatives (R4,200/review).

---

## Business Impact Summary

| Metric | Value |
|---|---|
| Value per caught negative review | R850 (enables intervention) |
| Cost per missed negative review | R4,200 (delayed action + churn) |
| Churn cost per customer | R8,500 × 8% probability |
| FN:FP cost ratio | ~11:1 |
| Annual reviews processed | 500,000 (2,000/day × 250 days) |
| DL advantage over keyword baseline | R15K–R80K per test batch |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `tensorflow` | Data pipelines, Tokenizer, pad_sequences, LSTM, Conv1D, BiLSTM |
| `pyspark` | Distributed data loading, null checks, sentiment labelling, EDA |
| `nltk` | Stopword list for text cleaning |
| `matplotlib.pyplot` | Training curves, distribution charts, heatmaps, ROC curves |
| `numpy` | Array operations, embedding matrix construction |
| `os` / `pathlib` | Directory management, file path handling |
| `sklearn` | Precision, recall, F1, ROC-AUC, confusion matrix, train_test_split |
| `seaborn` | Confusion matrix heatmaps |

---

## References

- McAuley, J. et al. (2015). *Image-based recommendations on styles and substitutes.* SIGIR 2015.
- Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014. https://nlp.stanford.edu/projects/glove/
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014.
- Schuster, M. & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks.* IEEE Transactions on Signal Processing.
