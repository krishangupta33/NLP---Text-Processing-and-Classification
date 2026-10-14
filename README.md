# NLP — Text Processing and Classification

A beginner-friendly, end-to-end NLP project that performs **text preprocessing** on app reviews and builds a **binary sentiment classifier** (Positive / Negative) using **Logistic Regression** on top of four different text-vectorization schemes (Binary Bag-of-Words, Count Bag-of-Words, N-grams, and TF-IDF).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Overview](#2-business-overview)
3. [Aim](#3-aim)
4. [Data Description](#4-data-description)
5. [Tech Stack](#5-tech-stack)
6. [Approach / Solution Methodology](#6-approach--solution-methodology)
7. [Repository Structure](#7-repository-structure)
8. [Model Results Summary](#8-model-results-summary)
9. [Installation & Setup](#9-installation--setup)
10. [Usage](#10-usage)
11. [Project Takeaways — All 18 Answered in Detail](#11-project-takeaways--all-18-answered-in-detail)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Project Overview

Have you ever wondered how a machine deals with the text we write or the sentences we speak? How does it take essential decisions based on *just text*? **Natural Language Processing (NLP)** gives machines the ability to understand, read, and extract meaningful insights from human language.

NLP is prospering today because of the huge availability of data and computational power, and it has spread from healthcare, media, and finance to human resources. This project introduces NLP and its core techniques in a lucid, step-by-step manner:

- **Part 1 — Text Preprocessing:** lowercasing, tokenization, stopword removal, punctuation removal, and stemming (Porter & Lancaster).
- **Part 2 — Feature Engineering:** converting text to numbers using Binary BoW, Count BoW, N-grams (1–3), and TF-IDF.
- **Part 3 — Modeling:** training a Logistic Regression classifier, evaluating train/test accuracy, and predicting sentiment for brand-new reviews.

The project ships in **two forms**:

| Form | Location | Purpose |
|---|---|---|
| Research / learning notebook | `model.ipynb` (root) and `Project_Modular_Code/linear_model.ipynb` | Step-by-step exploration with visualizations, following the video walkthrough |
| Production-style modular code | `Project_Modular_Code/` | Config-driven, CLI-based training & inference pipeline (`Engine.py`, `predict.py`) |

## 2. Business Overview

Apps live and die by user feedback. Manually reading thousands of reviews to understand what users love or hate is expensive and slow. An automated **sentiment classifier** lets a product team:

- automatically label every incoming review as **Positive** or **Negative**,
- triage negative feedback (bugs, crashes, save failures) to the right team faster,
- track sentiment trends over releases,
- and quantify user satisfaction at scale.

This project builds exactly that: a lightweight, fast, and interpretable sentiment classifier trained on **1,500 real Canva app reviews** scraped from the Google Play Store.

## 3. Aim

> To understand basic text preprocessing and build a binary classification model on the processed text.

## 4. Data Description

**File:** `Canva_reviews.xlsx` (also at `Project_Modular_Code/Input/Canva_reviews.xlsx`)

The dataset contains **1,500 reviews** about the Canva design app along with their sentiment labels and various metadata columns.

| Column | Description |
|---|---|
| `reviewId` | Unique Google Play review ID |
| `userName` | Reviewer's display name |
| `userImage` | Reviewer's avatar URL |
| `review` | **The review text (model input)** |
| `score` | Star rating given by the user (1–5) |
| `thumbsUpCount` | Number of "helpful" votes the review received |
| `reviewCreatedVersion` | App version the review was written for |
| `at` | Timestamp of the review |
| `replyContent` | Developer's reply text (if any) |
| `repliedAt` | Timestamp of the developer reply |
| `Sentiment` | **Label: `Positive` or `Negative` (target)** |
| `Sub Category` / `Sub Category_test` | Issue-category annotations (not used by the model) |

**Class distribution:**

| Sentiment | Count | Share |
|---|---|---|
| Positive | 1,032 | 68.8% |
| Negative | 468 | 31.2% |

**Review length (characters) statistics:** mean ≈ 86.5, std ≈ 55.7, min = 20, max = 498 (median 75) — most reviews are short single-paragraph comments.

## 5. Tech Stack

- **Language:** Python 3.8
- **Libraries:** `pandas`, `scikit-learn`, `nltk`, `seaborn`, `matplotlib`, `openpyxl`
- **Serialization:** `pickle` (models + vectorizers saved to `Output/`)

Exact pinned versions are in [`Project_Modular_Code/requirements.txt`](Project_Modular_Code/requirements.txt).

## 6. Approach / Solution Methodology

```
Raw Excel reviews
      │
      ▼
┌──────────────────────┐
│ 1. Data exploration  │  class balance, review length, score-vs-sentiment plots
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Preprocessing     │  lowercase → word_tokenize → stopword removal
│    (NLTK)            │  → punctuation removal (RegexpTokenizer) → stemming (Porter)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Vectorization     │  Binary BoW | Count BoW | N-grams (1–3) | TF-IDF   (min_df=5)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Train/test split   │  80/20, stratified on Sentiment, random_state=42
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. Model training     │  Logistic Regression → train/test accuracy
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Save artifacts    │  vectorizer.pkl + model.pkl → Output/
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 7. Inference         │  predict.py → P(Positive) for any new review
└──────────────────────┘
```

## 7. Repository Structure

```
NLP---Text-Processing-and-Classification/
│
├── model.ipynb                       # Step-by-step learning notebook (exploration → predictions)
├── Canva_reviews.xlsx                 # Raw dataset (copy)
├── Solution Methdology.pdf            # Project brief: overview, approach, 18 takeaways
├── README.md                          # ← You are here
│
└── Project_Modular_Code/              # Production-style, config-driven pipeline
    ├── Engine.py                      # Main training entry point (CLI)
    ├── predict.py                     # Inference entry point (CLI)
    ├── config.py                      # All hyperparameters & paths in one place
    ├── requirements.txt               # Pinned dependencies
    ├── linear_model.ipynb             # Same notebook, packaged inside the module
    ├── README.md                      # Setup/run instructions for the modular code
    ├── Input/
    │   └── Canva_reviews.xlsx         # Training data
    ├── Output/                        # Pre-trained artifacts (pickle)
    │   ├── binary_count_vect(.pkl/_lr.pkl)      # Binary BoW vectorizer + model
    │   ├── count_vect(.pkl/_lr.pkl)             # Count BoW vectorizer + model
    │   ├── n_gram(.pkl/_lr.pkl)                 # N-gram vectorizer + model  (BEST)
    │   ├── tf-idf(.pkl/_lr.pkl)                 # TF-IDF vectorizer + model
    │   └── model(.pkl/_lr.pkl)                  # Default 'bow' run artifacts
    └── Source/
        ├── processing.py              # process_text(): full cleaning pipeline
        ├── model.py                   # vectorize(): build features + split data
        └── utils.py                    # save_file() / load_file() pickle helpers
```

**Modular code roles:**

- **`config.py`** — single source of truth: `rs=42` (seed), `stem='p'` (Porter), `min_df=5`, n-gram range `(1,3)`, `test_size=0.2`, column names, and `Input`/`Output` paths.
- **`Source/processing.py`** — `process_text(review, stem)` implements the full cleaning pipeline (lowercase → tokenize → stopwords → punctuation → stem) and returns a clean string.
- **`Source/model.py`** — `vectorize(...)` builds the chosen vectorizer, fits it, and returns stratified train/test splits.
- **`Source/utils.py`** — pickle save/load helpers.
- **`Engine.py`** — orchestrates: read Excel → preprocess → vectorize → train Logistic Regression → print accuracies → save vectorizer & model.
- **`predict.py`** — loads a saved vectorizer+model pair, cleans new text identically, and prints the probability of the Positive class.

## 8. Model Results Summary

All runs: `LogisticRegression()`, `train_test_split(test_size=0.2, stratify=y, random_state=42)`, `min_df=5`.

| # | Vectorization | Features | Train Accuracy | Test Accuracy |
|---|---|---|---|---|
| 1 | **Binary BoW** (`CountVectorizer(binary=True)`) | 413 | 96.08% | **89.00%** |
| 2 | Count BoW (`CountVectorizer`) | 413 | 95.67% | 88.33% |
| 3 | **N-grams (1–3)** (`CountVectorizer(ngram_range=(1,3))`) | 666 | 97.17% | **89.33%** ✅ Best |
| 4 | TF-IDF (`TfidfVectorizer`) | 413 | 93.50% | 87.67% |

**Key observations:**

- All four variants generalize well (small train–test gap of ~7–8 points, typical of a high-bias linear model on a small dataset).
- **N-grams (1–3) performs best (89.33% test)** — multi-word phrases like `pleas fix`, `nt work`, `great app`, `best app ever` carry more sentiment signal than isolated words.
- Binary BoW slightly beats plain counts: for short reviews, *whether* a word appears matters more than *how often* it repeats.
- TF-IDF trails slightly here — with `min_df=5` already pruning rare, low-IDF noise words, the extra reweighting adds little for a linear model on 1,500 short documents.

**Sanity-check predictions (using the saved N-gram model):**

| Test review | P(Positive) | P(Negative) | Predicted |
|---|---|---|---|
| "this is a truly amazing app, best for those who have content but don't know how to express it..." | **99.56%** | 0.44% | ✅ Positive (1) |
| "Its the worst app ever I save my design lts not save" | 12.22% | **87.78%** | ✅ Negative (0) |

## 9. Installation & Setup

### Option A — Quick setup with `uv` (recommended, Python 3.12)

The original pinned versions (Python 3.8, `pandas 1.3.0`, `scikit-learn 1.0`) are outdated and won't install on modern Python, but the code itself runs fine on current libraries. The quickest path uses the [`uv`](https://docs.astral.sh/uv/) package manager:

```bash
cd Project_Modular_Code

# 1. Create a virtual environment with Python 3.12
uv venv --python 3.12

# 2. Install dependencies (verified modern pins from requirements.txt)
uv pip install --python .venv/bin/python -r requirements.txt

# 3. Download the NLTK corpora (note: modern nltk also needs punkt_tab)
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet']]"

# 4. Retrain the models — this regenerates the pickles with YOUR sklearn
#    version (the pre-trained pickles in Output/ were saved with sklearn 1.0)
.venv/bin/python Engine.py --vectorizer bowb --output_name binary_count_vect
.venv/bin/python Engine.py --vectorizer bow  --output_name count_vect
.venv/bin/python Engine.py --vectorizer ng   --output_name n_gram
.venv/bin/python Engine.py --vectorizer tf   --output_name tf-idf

# 5. Predict on a new review
.venv/bin/python predict.py --text "Its the worst app ever I save my design lts not save" --model_name n_gram
# → Probability of Positive Class: 12.29  (i.e., predicted Negative)
```

Verified results with Python 3.12 + `nltk 3.10` / `pandas 3.0` / `scikit-learn 1.9` (the exact pins now in [`Project_Modular_Code/requirements.txt`](Project_Modular_Code/requirements.txt)):

| Vectorizer | Train | Test |
|---|---|---|
| Binary BoW | 96.08% | 89.33% |
| Count BoW | 95.58% | 88.33% |
| N-grams (1–3) | 97.08% | 89.33% |
| TF-IDF | 93.42% | 87.67% |

(Closely matches the original run; small differences come from library-version changes.)

**Two gotchas when using modern versions:**

1. **The pre-trained pickles in `Output/` were created with sklearn 1.0 / Python 3.8.** Loading them with a different sklearn version raises an `InconsistentVersionWarning` (or may fail) — just retrain with `Engine.py` as shown above; it takes seconds and overwrites them with version-matched pickles.
2. **The learning notebooks use `vect.get_feature_names()`**, which was removed in scikit-learn 1.2 (replaced by `get_feature_names_out()`). If you want to run `model.ipynb` / `linear_model.ipynb` on modern sklearn, replace those calls, or run the notebook in a Python 3.8 environment with `requirements.txt`.

### Option B — Original setup (Python 3.8, pinned versions)

**Python Version: 3.8.10** (the modular code was built and tested on Python 3.8).

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd NLP---Text-Processing-and-Classification/Project_Modular_Code
   ```

2. **Create & activate a virtual environment**
   - Windows:
     ```bat
     py -3.8 -m venv venv
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download the required NLTK corpora** — inside a Python shell or notebook:
   ```python
   import nltk
   nltk.download()   # opens the NLTK downloader GUI
   ```
   Install at minimum:
   - `punkt` (tokenizers)
   - `stopwords` (English stopword list)
   - `wordnet` (lexical resource)

   Or non-interactively:
   ```bash
   python -c "import nltk; [nltk.download(p) for p in ['punkt','stopwords','wordnet']]"
   ```

## 10. Usage

*(Commands below assume the environment is activated — or prefix each with `.venv/bin/python` as shown in Section 9.)*

### 10.1 Train a model

From inside `Project_Modular_Code/`:

```bash
python Engine.py --file_name Canva_reviews.xlsx --vectorizer bowb --output_name binary_count_vect
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--file_name` | `Canva_reviews.xlsx` | Excel file inside `Input/` |
| `--vectorizer` | `bow` | One of: `bow` (count BoW), `bowb` (binary BoW), `ng` (n-grams), `tf` (TF-IDF) |
| `--output_name` | `model` | Base name for saved artifacts → `Output/<name>.pkl` (vectorizer) and `Output/<name>_lr.pkl` (model) |

**Example — train all four variants:**

```bash
python Engine.py --vectorizer bowb --output_name binary_count_vect
python Engine.py --vectorizer bow  --output_name count_vect
python Engine.py --vectorizer ng   --output_name n_gram
python Engine.py --vectorizer tf   --output_name tf-idf
```

Each run prints train/test accuracy and writes the vectorizer + model pickles to `Output/`.

### 10.2 Predict on a new review

```bash
python predict.py --text "Its the worst app ever I save my design lts not save" --model_name binary_count_vect
```

- `--model_name` is the **same base name used at training time** (default `n_gram`, the best model).
- The script cleans the input through the identical preprocessing pipeline, vectorizes it with the saved vectorizer, and prints the **probability of the Positive class**.

### 10.3 Use the notebook instead

Open `model.ipynb` (root) or `Project_Modular_Code/linear_model.ipynb` in Jupyter and run top-to-bottom to reproduce the full exploration → preprocessing → 4 vectorizers → predictions walkthrough with plots.

---

## 11. Project Takeaways — All 18 Answered in Detail

These are the 18 takeaways defined in `Solution Methdology.pdf`, answered with reference to what was actually implemented in this repository.

---

### 1. Understanding the problem statement and the approach

**Problem:** Given free-form app-review *text*, predict whether the sentiment is **Positive (1)** or **Negative (0)**.

**Why it's non-trivial:** ML models consume numeric vectors, not strings. Reviews vary in length, casing, punctuation, and vocabulary, and carry noise (stopwords, typos, emoticons like `:(`).

**The approach chosen:**

1. **Explore** the data — class balance, review lengths, score-vs-sentiment relationship.
2. **Preprocess** every review into a normalized bag of clean word-stems (lowercase → tokenize → remove stopwords → remove punctuation → stem).
3. **Vectorize** the cleaned text into fixed-length numeric vectors using four schemes (Binary BoW, Count BoW, N-grams, TF-IDF) — all restricted to words appearing in ≥ 5 documents (`min_df=5`) to control dimensionality.
4. **Train** a Logistic Regression classifier on an 80/20 stratified split.
5. **Compare** test accuracies, keep the best (N-grams, 89.33%), save artifacts, and serve predictions on new text.

**Key design decisions:** stratified splitting (preserves the 68.8/31.2 class ratio in train and test), a fixed `random_state=42` for reproducibility, identical preprocessing at train and inference time (encapsulated in `process_text()`), and pickling the vectorizer *together with* the model so both sides of the pipeline always match.

---

### 2. Data exploration and visualization

Performed in `model.ipynb` on the 1,500 × 13 dataset:

- **Class balance** — `sns.countplot(x="Sentiment")` and `value_counts()`: 1,032 Positive vs 468 Negative → 31.2% negative, 68.8% positive. The dataset is **imbalanced but not severely**, so plain accuracy is still a reasonable metric.
- **Star score distribution** — `sns.countplot(x="score")` shows ratings cluster at the extremes (mostly 5s and 1s).
- **Score–sentiment relationship** — `sns.countplot(x="score", hue="Sentiment")` confirms intuition: low star ratings are dominated by Negative labels and high ratings by Positive — but not perfectly (e.g., a 4-star review was labeled Negative because it complained about picture quality). This is why the model is built on the **review text**, not the score.
- **Review length** — a new `len` column (`data["review"].apply(len)`) and `describe()` give: mean ≈ 86.5 chars, std ≈ 55.7, min 20, max 498.
- **Length distribution & by-class KDE** — `sns.displot(data["len"])` shows a right-skewed histogram; `sns.displot(..., hue="Sentiment", kind="kde", fill=True)` shows Positive and Negative reviews have overlapping length distributions, so **length alone is not discriminative** — again motivating word-level features.

---

### 3. What is tokenization?

**Tokenization** is the process of breaking a stream of text into smaller units called **tokens** (words, punctuation marks, symbols). It is the first structural step of any NLP pipeline because all downstream operations (stopword removal, counting, stemming, vectorization) operate on *tokens*, not raw strings.

**Levels of tokenization:**

- **Sentence tokenization** — splitting a document into sentences (boundaries like `.`, `!`, `?`).
- **Word tokenization** — splitting a sentence into words and punctuation symbols.

**Example from the notebook:**

```python
review = "Unable to save my work. Nothing works :("

sent_tokenize(review)
# ['Unable to save my work.', 'Nothing works :(']

word_tokenize(review)
# ['Unable', 'to', 'save', 'my', 'work', '.', 'Nothing', 'works', ':', '(']
```

Note that `word_tokenize` keeps punctuation as separate tokens (`.`, `:`, `(`) — those are removed in a later step.

---

### 4. Performing tokenization using `word_tokenize` from the NLTK library

The project uses NLTK's **`word_tokenize`** (backed by the TreebankWordTokenizer, requires the `punkt` corpus) applied **after lowercasing** the whole corpus:

```python
# lowercase all 1500 reviews first
reviews_lower = [r.lower() for r in reviews]

# tokenize every review
tokens = [word_tokenize(r) for r in reviews_lower]

tokens[13]
# ['unable', 'to', 'save', 'my', 'work', '.', 'nothing', 'works', ':', '(']

tokens[1495]
# ['absolutely', 'amazing', 'and', 'a', 'lifesaver', 'for', 'teachers', '.']
```

**Why lowercase *before* tokenizing:** otherwise `"Unable"` and `"unable"` would become different tokens and later different vocabulary entries; normalizing case collapses them into one feature.

**Order of operations matters:** lowercase → tokenize → *then* filter (stopwords/punctuation) → *then* stem. In the modular code this exact chain lives in `Source/processing.py::process_text()`:

```python
review = review.lower()
tokens = word_tokenize(review)
```

---

### 5. What are stopwords?

**Stopwords** are extremely frequent, function words that carry almost no sentiment or topical information — articles, pronouns, prepositions, conjunctions (`the`, `is`, `at`, `which`, `on`, `to`, `my`, ...).

**Why remove them?**

- They **dominate raw word counts** (a huge share of the 13,782 corpus tokens), inflating dimensionality and diluting the informative words.
- They add **noise, not signal** — `"to"` tells you nothing about whether a review is positive or negative.
- Removing them shrinks vectors and speeds up training, with no loss of discriminative power for this task.

NLTK ships ready-made lists (requires the `stopwords` corpus):

```python
from nltk.corpus import stopwords
sw = stopwords.words('english')
sw[:10]
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
```

**Caveat demonstrated by the data:** negations like `"not"`/`"n't"` are also in many stopword lists. In this pipeline `word_tokenize` splits `"don't"` → `do` + `n't` and stopword removal deletes `do`, leaving `nt` — which the model actually uses in strong negative features like `nt work` / `nt save` (visible in the n-gram vocabulary). So blind stopword removal can be risky in general, but here the surviving `nt` token conveniently keeps negation information.

---

### 6. How to remove stopwords?

A **list-comprehension filter** over the token lists, keeping only tokens *not* in the stopword set:

```python
sw = stopwords.words('english')
tokens = [[word for word in t if word not in sw] for t in tokens]

tokens[13]
# ['unable', 'save', 'work', '.', 'nothing', 'works', ':', '(']      # 'to', 'my' gone
tokens[1495]
# ['absolutely', 'amazing', 'lifesaver', 'teachers', '.']           # 'and', 'a', 'for' gone
```

In the modular code (`Source/processing.py`):

```python
sw = stopwords.words('english')           # module level, loaded once
tokens = [t for t in tokens if t not in sw]
```

**Practical notes:**

- Membership testing against a **Python list** is O(n) per lookup — for large corpora convert it to a **set** (`sw = set(stopwords.words('english'))`) for O(1) checks.
- Always remove stopwords **in the same language** as the text and **the same way at inference time** — `predict.py` reuses the identical filter, otherwise train/test feature distributions mismatch.

---

### 7. Removing the punctuations

Punctuation (`.`, `,`, `!`, `(`, `:`, ...) survives `word_tokenize` as standalone tokens. It is removed using NLTK's **`RegexpTokenizer`** with the pattern `r'\w+'`, which matches only alphanumeric character runs:

```python
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

tokenizer.tokenize("wasn't")   # ['wasn', 't']   → rejoined: "wasnt"
tokenizer.tokenize(":")        # []              → pure punctuation yields nothing
```

The project's trick is to run each token through the regex tokenizer and **join the pieces back**, then keep only non-empty results:

```python
tokens = [["".join(tokenizer.tokenize(word)) for word in t
           if len(tokenizer.tokenize(word)) > 0] for t in tokens]

tokens[13]
# ['unable', 'save', 'work', 'nothing', 'works']     # '.', ':', '(' all removed
```

**Effects:**

- Pure punctuation tokens (`.` `:` `(`) produce empty lists → dropped entirely.
- Contractions get **glued back together**: `"wasn't"` → `"wasnt"` (one token instead of losing the negation entirely).

**Trade-off to know:** this discards potentially useful sentiment cues such as `!!!`, `:(`, `:)`, and ALL-CAPS emphasis. A more advanced pipeline would keep emoticons as special tokens; here simplicity wins and the model still reaches ~89%.

---

### 8. What is stemming?

**Stemming** is the crude, rule-based chopping of word **inflections** down to a common **stem**, so that `"works"`, `"working"`, `"worked"`, and `"worker"` all map to the same feature (`work`).

**Why do it?**

- Without stemming, `love` / `loved` / `loving` become **three separate, sparse features** — each seen in fewer documents, so the model has less evidence per feature.
- With stemming they collapse into one high-count feature, **densifying** the data and improving generalization on unseen word forms.

**Stemming vs. lemmatization:** stemming is heuristic string surgery (fast, sometimes over- or under-chops); lemmatization uses a dictionary to return proper root words ("better" → "good"), which is more accurate but slower and needs POS info. This project uses **stemming** for speed and simplicity.

Examples (Porter):

```python
PorterStemmer().stem("teachers")     # 'teacher'
PorterStemmer().stem("absolutely")   # 'absolut'
```

---

### 9. Stemming using Porter Stemmer

The **Porter Stemmer** (1980) is the classic suffix-stripping algorithm: it applies a fixed cascade of rules (plural removal, `-ing`/`-ed` removal, terminal `-e` deletion, etc.) in five phases.

Demonstration in the notebook:

```python
porter = PorterStemmer()
porter.stem("teachers")     # 'teacher'   (plural removed, word stays readable)
porter.stem("absolutely")   # 'absolut'   (adverb suffix stripped)
```

Applied to the whole corpus:

```python
tokens = [[porter.stem(word) for word in t] for t in tokens]

tokens[13]
# ['unabl', 'save', 'work', 'noth', 'work']
tokens[1495]
# ['absolut', 'amaz', 'lifesav', 'teacher']
```

Notice `"Unable"` → `unabl`, `"amazing"` → `amaz`, `"lifesaver"` → `lifesav`, `"teachers"` → `teacher`. All four reviews that said *use / used / using* now share the single feature `use` (474 occurrences in the corpus). This is the stemmer actually doing feature consolidation — and it's **the default in this project** (`config.py`: `stem = "p"`).

---

### 10. Stemming using Lancaster Stemmer

The **Lancaster (Paice/Husk) Stemmer** is a more aggressive, iterative rule-based stemmer with a large rule table that keeps applying rules until no more match.

**Side-by-side from the notebook:**

```python
lancaster = LancasterStemmer()

porter.stem("teachers")      # 'teacher'
lancaster.stem("teachers")   # 'teach'        ← heavier chopping

porter.stem("absolutely")    # 'absolut'
lancaster.stem("absolutely") # 'absolv'       ← less readable / can over-stem
```

**Porter vs. Lancaster:**

| Aspect | Porter | Lancaster |
|---|---|---|
| Aggressiveness | Moderate | Very aggressive |
| Readability of stems | Better (`teacher`) | Often cryptic (`teach`, `absolv`) |
| Speed | Slower on huge corpora | Fast, iterative rule table |
| Risk | Mild under-stemming | Over-stemming — distinct words may collide |

The pipeline supports **both** via a parameter (`stem='p'` or `'l'` in `process_text()`), with Porter as the default because milder stems preserve more distinction between words on a small dataset.

---

### 11. Distributions of words in each document

After preprocessing, the corpus was flattened and counted to understand word usage:

```python
flat_tokens = [word for t in tokens for word in t]
len(flat_tokens)          # 13,782 total tokens
counts = Counter(flat_tokens)
len(counts)              # 1,720 unique words
counts.most_common(10)
# [('app', 932), ('use', 474), ('design', 349), ('easi', 293),
#  ('s', 240), ('love', 220), ('good', 219), ('edit', 191),
#  ('make', 190), ('great', 184)]
```

**Interpretation:**

- **1,720 unique stems across 13,782 tokens** ≈ 8 occurrences per word on average — but the distribution is extremely **Zipfian (long-tailed)**: `app` alone appears 932 times while hundreds of words appear only once or twice.
- Sentiment-bearing words are prominent: `love`, `good`, `great`, `easi` (positive-leaning) vs `unabl` / `nt` / `noth` (negative-leaning), often co-occurring with issue words like `save`, `download`, `crash`.
- This long tail is exactly why **`min_df=5`** is used at vectorization: it prunes hapax words, shrinking the vocabulary from 1,720 to **413** (unigrams) — a 4× smaller, far less overfittable feature space.

---

### 12. Creating Bag of Words with binary

**Bag-of-Words (BoW)** represents a document by its words, ignoring order — a document becomes a vector of per-vocabulary-word statistics.

**Binary BoW** records only **presence/absence (1/0)** of each vocabulary word in a document:

```python
vect = CountVectorizer(binary=True, min_df=5)
X = vect.fit_transform(clean_reviews)   # sparse CSR matrix 1500 × 413
X_a = X.toarray()

clean_reviews[13]                # 'unabl save work noth work'
X_a[13, :].sum()                 # 4  → 4 distinct vocab words present
X_a[13, vect.get_feature_names().index("work")]   # 1 (appears once → stored as 1)
```

**Result with Logistic Regression:** Train **96.08%**, Test **89.00%** — the best *unigram* variant of the four.

**Why binary works well here:** reviews are short (mean ~86 chars), so repeated words are rare; *whether* a user said `worst` matters much more than saying it twice. Binary features also behave like one-hot indicators, which suits a linear model.

---

### 13. Creating Bag of Words without binary

Plain **count BoW** stores the **frequency** of every vocabulary word in the document:

```python
vect = CountVectorizer(min_df=5)
X = vect.fit_transform(clean_reviews)

X_a[13, vect.get_feature_names().index("work")]   # 2  → 'work' occurs twice in review 13
```

Notice the same review 13 (`unabl save work noth work`): binary stored 1 for `work`, counts store **2** — that single cell is the only difference between the two schemes on this corpus.

**Result with Logistic Regression:** Train **95.67%**, Test **88.33%** — slightly below binary (89.00%).

**Interpretation:** on short texts, raw counts add little information and can let one repeated word dominate the dot-product inside the linear model; presence-flags generalize marginally better.

---

### 14. Creating Bag of Words with N-grams

An **n-gram** is a contiguous sequence of *n* tokens. `ngram_range=(1, 3)` builds unigrams **+** bigrams **+** trigrams as features, so the vocabulary includes `pleas fix`, `nt work`, `great app`, `best app ever`, etc.

```python
vect = CountVectorizer(min_df=5, ngram_range=(1, 3))
X = vect.fit_transform(clean_reviews)
X.shape          # (1500, 666) — 413 unigrams + frequent bigrams/trigrams = 666 features
```

**Why n-grams help:** they restore a little of the word **order** that BoW throws away. `not good` (negative) vs `good` (positive) is distinguishable because `nt good`/`not good` becomes its own feature. Strong features learned include `pleas fix`, `ca nt`, `take much time`, `best app ever`, `great app love`.

**Result with Logistic Regression:** Train **97.17%**, Test **89.33%** — **the best model in the project**, saved as `Output/n_gram.pkl` + `n_gram_lr.pkl` and used as the default in `predict.py`.

**Cost:** the feature space grows (413 → 666 here; far more on larger corpora), so n-grams need `min_df` pruning and work best when the dataset isn't tiny.

---

### 15. What is TF-IDF?

**Term Frequency — Inverse Document Frequency** reweights BoW counts so that:

- **TF** — how often a word occurs *in a document* (a word used a lot in one review is important *for that review*), and
- **IDF** — how **rare** the word is across *all documents* (a word appearing in every review, like `app`, is not discriminative; a word appearing in a handful, like `worst`, is).

**Formula (sklearn's smooth scheme):**

$$idf(t) = \ln\frac{1 + N}{1 + df(t)} + 1, \qquad tfidf(t,d) = tf(t,d) \times idf(t)$$

followed by L2 row-normalization. Frequent-but-common words (`app`, `use`, `design`) get **down-weighted**; rare-but-focused words get **boosted** — effectively an automatic, corpus-level version of `min_df`.

```python
vect = TfidfVectorizer(min_df=5)
X = vect.fit_transform(clean_reviews)   # (1500, 413)
```

---

### 16. How to find TF-IDF score?

**With sklearn** — fit `TfidfVectorizer` and transform (same API as `CountVectorizer`, so the rest of the pipeline is unchanged):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5)
X = vect.fit_transform(clean_reviews)     # each row is the TF-IDF vector for one review
```

**By hand** — for a term $t$ in document $d$:

1. $tf(t,d)$ = count of $t$ in $d$ (or count / total words in $d$).
2. $df(t)$ = number of documents containing $t$.
3. $idf(t) = \ln\frac{N}{df(t)}$ (variants add smoothing/`+1`).
4. $tfidf(t,d) = tf(t,d) \times idf(t)$; typically L2-normalize the row.

**Result with Logistic Regression:** Train **93.50%**, Test **87.67%** — the weakest of the four here. Why? The reviews are short and `min_df=5` already removed low-value rare words, so there is little common-word inflation left for IDF to fix; and the real signal lives in *phrases*, which only the n-gram model captures.

---

### 17. Building the Logistic Regression model

**Why Logistic Regression for text?** It is fast, interpretable (signed per-feature weights = positive/negative evidence), works naturally with high-dimensional sparse matrices, and is a strong baseline for BoW/TF-IDF text classification.

**Setup** (identical for every vectorizer):

```python
# labels: Positive → 1, Negative → 0
data["Sentiment"] = data["Sentiment"].apply(lambda x: 1 if x == "Positive" else 0)
y = data["Sentiment"]

# stratified 80/20 split, fixed seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# train
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)
print(accuracy_score(y_train, train_pred))   # e.g. 0.9717 (n-grams)
print(accuracy_score(y_test,  test_pred))    # e.g. 0.8933 (n-grams)
```

**How it works:** for a feature vector $x$, the model computes $P(\text{Positive}) = \sigma(w^\top x + b)$ with $\sigma(z) = 1/(1+e^{-z})$ and is optimized by (regularized) log-loss via gradient descent (LBFGS in sklearn). A large positive weight on `great app` pushes the prediction toward 1; a large negative weight on `nt work` pushes it toward 0.

**Results recap (all with default `LogisticRegression()`):**

| Vectorizer | Train | Test |
|---|---|---|
| Binary BoW | 96.08% | 89.00% |
| Count BoW | 95.67% | 88.33% |
| N-grams (1–3) | 97.17% | **89.33%** |
| TF-IDF | 93.50% | 87.67% |

The modest train–test gap shows a **good bias/variance balance** — the linear model is too simple to overfit badly even with 666 sparse features. Both the vectorizer **and** model are pickled so the exact pipeline can be restored later.

---

### 18. Predictions on new reviews

**End-to-end inference** — the saved **N-gram** vectorizer + model are loaded, and the *exact same preprocessing* (lowercase → tokenize → stopwords → punctuation → Porter stem) is applied before `vect.transform`:

```python
vect  = pickle.load(open("Output/n_gram.pkl", "rb"))
model = pickle.load(open("Output/n_gram_lr.pkl", "rb"))

# clean the new text through the SAME pipeline...
clean_review = [" ".join(porter.stem(w) for w in ...)]   # (full chain, see notebook)

# ...then transform and predict
X_new = vect.transform(clean_review)     # (1, 666) — same columns as training
model.predict_proba(X_new)               # class probabilities
model.predict(X_new)                     # hard label
```

**Results from the notebook:**

| Review | `predict_proba` | `predict` |
|---|---|---|
| "this is a truly amazing app, best for those who have content but don't know how to express it in a good and shareable manner. Thanks Team Canva..." | `[[0.0044, 0.9956]]` → **99.56% positive** | `1` ✅ |
| "Its the worst app ever I save my design lts not save" | `[[0.8778, 0.1222]]` → **87.78% negative** | `0` ✅ |

Note that the model doesn't just output a label — the **probability** (`predict_proba`) quantifies confidence, which is what `predict.py` surfaces to the user.

**CLI equivalent:**

```bash
python predict.py \
  --text "Its the worst app ever I save my design lts not save" \
  --model_name n_gram
# → Probability of Positive Class: 12.22  (i.e., predicted Negative)
```

The critical engineering point: **the vectorizer used at inference must be the very one fit during training** (same vocabulary, same column order, same n-gram ranges), and the text must be cleaned by the *identical* `process_text()` function — otherwise the feature vector is meaningless.

---

## 12. Limitations & Future Work

**Limitations**

- Small dataset (1,500 reviews) from a single app — sentiment vocabulary may not transfer to other domains.
- Accuracy-only evaluation; on this 68.8/31.2 split, a naive majority classifier already scores 68.8%. (Precision/recall/F1, a confusion matrix, and ROC-AUC would be more informative.)
- Aggressive cleaning drops emoticons (`:(`, `:)`) and emphasis that carry sentiment.
- Porter stemming sometimes produces non-word stems (`amaz`, `lifesav`) that hurt interpretability.

**Future work**

- Report F1 / confusion matrix / ROC-AUC; tune `LogisticRegression` C and `min_df` / `ngram_range` via cross-validation.
- Try Naive Bayes, linear SVM, and tree ensembles; compare with word embeddings (word2vec/GloVe) or a fine-tuned transformer (BERT/DistilBERT).
- Lemmatization instead of stemming; keep negations explicitly (`not_good`); class-weighting or SMOTE for imbalance.
- Expand the dataset to multiple apps/languages and package the pipeline as a REST API (FastAPI/Flask) for live inference.

---

*Pre-trained vectorizers and models are included in `Project_Modular_Code/Output/` — you can run `predict.py` immediately without retraining. See [`Project_Modular_Code/README.md`](Project_Modular_Code/README.md) for the original setup instructions.*
