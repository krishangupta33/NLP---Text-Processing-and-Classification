# NLP — Text Processing and Classification

Predict whether an app review is **Positive** or **Negative** using Python and machine learning.

This project takes 1,500 real Canva app reviews, cleans the text, converts it to numbers, and trains a Logistic Regression classifier. It compares **4 different ways** of turning text into numbers and shows how to predict sentiment for any new review.

---

## How It Works (5 steps)

```
1. Read reviews     →  Canva_reviews.xlsx (1,500 reviews)
2. Clean the text   →  lowercase, remove stopwords & punctuation, stem words
3. Text → numbers   →  one of: Binary BoW | Count BoW | N-grams | TF-IDF
4. Train            →  Logistic Regression (80% train / 20% test)
5. Predict          →  new review in → "Probability of Positive: 98.64" out
```

---

## Results

| Method | Test Accuracy |
|---|---|
| Binary Bag-of-Words | 89.00% |
| Count Bag-of-Words | 88.33% |
| **N-grams (best)** | **89.33%** |
| TF-IDF | 87.67% |

All four work about equally well — N-grams wins because word pairs like *"not working"* carry more meaning than single words.

---

## Quick Start

You need [Python 3.12](https://www.python.org/) and [uv](https://docs.astral.sh/uv/) (or plain pip — see below).

```bash
cd Project_Modular_Code

# 1. Create a virtual environment
uv venv --python 3.12

# 2. Install the packages
uv pip install --python .venv/bin/python -r requirements.txt

# 3. Download the NLTK data (one time only)
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet']]"

# 4. Train the models
.venv/bin/python Engine.py --vectorizer bowb --output_name binary_count_vect
.venv/bin/python Engine.py --vectorizer bow  --output_name count_vect
.venv/bin/python Engine.py --vectorizer ng   --output_name n_gram
.venv/bin/python Engine.py --vectorizer tf   --output_name tf-idf
```

<details>
<summary>No uv? Use plain pip instead</summary>

```bash
cd Project_Modular_Code
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet']]"
.venv/bin/python Engine.py --vectorizer ng --output_name n_gram
```
</details>

Each training run prints the accuracy, e.g.:

```
Train Accuracy: 97.08%
Test Accuracy: 89.33%
```

---

## Predict a New Review

```bash
.venv/bin/python predict.py --text "Its the worst app ever I save my design lts not save" --model_name n_gram
```

Output:

```
Text: Its the worst app ever I save my design lts not save
Probability of Positive Class: 12.29
```

Low probability = negative review. Try it with your own text! (The model must be trained first — step 4 above.)

---

## Project Files

```
Project_Modular_Code/
├── Engine.py          # Train a model (run this)
├── predict.py         # Predict on new text (run this)
├── config.py          # Settings (seed, test size, columns, paths)
├── requirements.txt   # Package versions
├── linear_model.ipynb # Step-by-step learning notebook
├── Input/
│   └── Canva_reviews.xlsx   # The dataset
├── Output/            # Trained models saved here (.pkl files)
└── Source/
    ├── processing.py  # Text cleaning
    ├── model.py       # Text → numbers + train/test split
    └── utils.py       # Save/load helpers
```

---

## Notebook

The learning notebook is `Project_Modular_Code/linear_model.ipynb` — it walks through every step with plots and explanations. Run it from inside `Project_Modular_Code/` (the file paths in it are relative to that folder).

```bash
cd Project_Modular_Code
source .venv/bin/activate
jupyter lab        # or: .venv/bin/python -m jupyter lab
```

> Tip: skip the last cell (`nltk.download()`) — it opens a GUI window, and the data is already downloaded in step 3 above.

---

## The Dataset

`Canva_reviews.xlsx` — 1,500 Google Play reviews of the Canva design app.

| Sentiment | Count |
|---|---|
| Positive | 1,032 (68.8%) |
| Negative | 468 (31.2%) |

Only two columns matter for the model: `review` (the text) and `Sentiment` (the label).

---

## Want More Detail?

- 📚 **[PROJECT_TAKEAWAYS.md](PROJECT_TAKEAWAYS.md)** — all 18 concepts explained in depth (tokenization, stopwords, stemming, BoW, N-grams, TF-IDF, Logistic Regression)
- 📄 `Solution Methdology.pdf` — the original project brief
- 📖 `Project_Modular_Code/README.md` — original setup instructions (Python 3.8)

## Troubleshooting

| Problem | Fix |
|---|---|
| `LookupError: no resource ... punkt` | Re-run step 3 (NLTK download) |
| Error loading `.pkl` files / sklearn version warning | Retrain: run step 4 again (pickles must match your installed sklearn version) |
| `FileNotFoundError: Input/...` | Run from inside `Project_Modular_Code/`, not the repo root |
