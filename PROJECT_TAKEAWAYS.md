# Project Takeaways — All 18 Answered in Detail

These are the 18 takeaways defined in `Solution Methdology.pdf`, answered with reference to what is actually implemented in this repository (notebook: `Project_Modular_Code/linear_model.ipynb`, pipeline: `Engine.py` / `predict.py` / `Source/`).

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

Performed in the notebook on the 1,500 × 13 dataset:

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
# ['absolutely', 'amazing', 'lifesaver', 'teachers', '.']             # 'and', 'a', 'for' gone
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

Notice `"Unable"` → `unabl`, `"amazing"` → `amaz`, `"lifesaver"` → `lifesav`, `"teachers"` → `teacher`. All reviews that said *use / used / using* now share the single feature `use` (474 occurrences in the corpus). This is the stemmer actually doing feature consolidation — and it's **the default in this project** (`config.py`: `stem = "p"`).

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
X_a[13, vect.get_feature_names_out().tolist().index("work")]   # 1 (appears once → stored as 1)
```

**Result with Logistic Regression:** Train **96.08%**, Test **89.00%** — the best *unigram* variant of the four.

**Why binary works well here:** reviews are short (mean ~86 chars), so repeated words are rare; *whether* a user said `worst` matters much more than saying it twice. Binary features also behave like one-hot indicators, which suits a linear model.

---

### 13. Creating Bag of Words without binary

Plain **count BoW** stores the **frequency** of every vocabulary word in the document:

```python
vect = CountVectorizer(min_df=5)
X = vect.fit_transform(clean_reviews)

X_a[13, vect.get_feature_names_out().tolist().index("work")]   # 2  → 'work' occurs twice in review 13
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
python predict.py --text "Its the worst app ever I save my design lts not save" --model_name n_gram
# → Probability of Positive Class: 12.29  (i.e., predicted Negative)
```

The critical engineering point: **the vectorizer used at inference must be the very one fit during training** (same vocabulary, same column order, same n-gram ranges), and the text must be cleaned by the *identical* `process_text()` function — otherwise the feature vector is meaningless.
