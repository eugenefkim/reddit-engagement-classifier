# reddit-engagement-classifier

Distributed PySpark ML pipeline that classifies Reddit post engagement archetypes (viral, crowd-pleaser, debate-starter, low-engagement) using NLP feature engineering on 89GB of Pushshift Reddit data.

## Dataset

**Source:** [Pushshift Reddit Submissions Dataset](https://huggingface.co/datasets/fddemarco/pushshift-reddit)  
**Host:** HuggingFace (`fddemarco/pushshift-reddit`)  
**Format:** Parquet  
**Size:** ~89 GB, 549+ million rows  

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique post identifier |
| `author` | string | Reddit username of poster |
| `subreddit` | string | Community the post was submitted to |
| `subreddit_id` | string | Unique subreddit identifier |
| `title` | string | Post title text |
| `selftext` | string | Post body text (empty for link posts) |
| `score` | long | Net upvotes at time of archival |
| `num_comments` | long | Comment count at time of archival |
| `created_utc` | long | Unix timestamp of post creation |

## Environment Setup

### Cluster and Repository Access
All work for this repo was performed on the SDSC Expanse HPC cluster. Configuration settings must be set based on environment. 

1. Log into the Expanse. 
2. Clone the repository:
```bash
git clone https://github.com/eugenefkim/reddit-engagement-classifier.git
cd reddit-engagement-classifier
```

### Data Access

> **Note:** The module versions configured in the data load instructions below (`cpu/0.15.4`, `gcc/10.2.0`) are specific to the SDSC Expanse cluster Python 3.6 settings. Data loading instructions must be tailored based on working environment.

The dataset is hosted on HuggingFace and will be downloaded directly to the cluster for this repository structure. Raw data files are gitignored.

To download the data, open and run the [`notebooks/data_download.ipynb`](notebooks/data_download.ipynb) notebook sequentally from your JupyterLab session on Expanse, following the notebook instructions. This must be run on a compute node (not the login node) to avoid memory limitations, and should be run using Jupyter Notebook settings. 

The dataset is downloaded into `data/raw/`, approximately 89GB of 218 Parquet files. The notebook loads data directly from `data/raw/` using a relative path.


### SDSC Expanse Setup and SparkSession Configuration (Milestone 2)

#### Milestone 2 SparkSession Configuration 

The Expanse JupyterLab portal launches Spark inside a Singularity container which defaults to `local[*]` mode. This means there is no separate cluster manager and no worker processes and all driver coordination and task execution run in a single JVM, using the node's 16 cores as parallel thread slots. Parallelism is still achieved, but it is thread-based within one JVM rather than distributed across separate executor processes. As a result, executor configuration parameters are set but not active, and the Spark UI shows a single executor row. We also allocate 8GB to driver memory account for this.

```python
from pyspark.sql import SparkSession

# 16 cores, 128GB total memory (local mode)
spark = SparkSession.builder \
    .appName("PushshiftRedditEDA") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.instances", "6") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.parquet.enableVectorizedReader", "true") \
    .getOrCreate()
```

#### SparkUI Screenshot (Milestone 2)
<img width="853" height="423" alt="SparkUI_Screenshot_Local" src="https://github.com/user-attachments/assets/42f747a7-9c31-4201-8586-89919649a849" />

## Notebook (Milestone 2)

The Milestone 2 EDA and data exploration notebook is located at: [`notebooks/Milestone2_Pushshift.ipynb`](notebooks/Milestone2_Pushshift.ipynb)

## Milestone 2 - Exploratory Data Analysis

### How many observations does your dataset have? 
The dataset contains 549,662,955 posts. 

### Describe all columns in your dataset: their scales and data distributions. Describe categorical and continuous variables. Describe your target column.
Dataset descriptions were provided above in the `Dataset` section but are described in further detail below to better address Milestone 2 requirements.

#### Categorical Columns

- **`id`** (string): Unique post identifier. Approximately 550M distinct values with only 1 duplicate found across the full dataset, confirming near-perfect uniqueness. Not used as a model feature.
- **`author`** (string): Reddit username of the poster. ~35.4M distinct users. Heavily skewed the top authors are bots and aggregator accounts (`AutoNewsAdmin`, `AutoNewspaperAdmin`, `politicbot`, etc.) responsible for millions of posts each. ~13.3% of posts have `[deleted]` authors and ~5.5% have empty string authors. Not directly encoded as a feature; represented through author history aggregations.
- **`subreddit`** (string): Community the post was submitted to. ~2.3M distinct subreddits. Extremely long-tailed r/AskReddit alone accounts for ~13.5M posts while the vast majority of subreddits have very few. Not directly encoded; represented through per-subreddit aggregation features.
- **`subreddit_id`** (string): Internal Reddit identifier for the subreddit. ~2.2M distinct values, closely tracking `subreddit` cardinality. Used for deduplication and join purposes only.
- **`selftext`** (string): Post body text. 72.8% of posts have empty selftext, indicating link or image posts with no body. The remaining 27.2% are text posts. No null or sentinel string values present in this version of the dataset absence is encoded as an empty string. Represented as a binary `is_text_post` flag and selftext character length in modeling.
- **`title`** (string): Post title text. Present for all 549M rows. Mean length ~50 characters, std dev ~37, max 1,592 characters. Used as the primary source of NLP features (word count, sentiment, readability, question/exclamation presence).

#### Continuous Columns

- **`score`** (long): Net upvotes at time of archival. Range: 0 to 270,469. Mean: 44.84, std dev: 707.38 extremely right-skewed. The vast majority of posts score in the single digits while a small number go viral. Median is consistently far below mean in every year. One of the two **target label inputs**.
- **`num_comments`** (long): Comment count at time of archival. Range: -117 to 517,003. Mean: 8.36, std dev: 93.11 similarly right-skewed. Negative values are a known Pushshift artifact and will be treated as 0. The other **target label input**.
- **`created_utc`** (long): Unix timestamp of post creation. Spans January 1, 2012 to December 31, 2018. Used to extract temporal features (hour of day, day of week) and for train/test splitting. Post volume shows a clear weekday peak (Mon–Thu) and grows steadily year over year across the dataset's coverage window.

#### Target Column

The target label is not a raw column but is derived from the joint distribution of `score` and `num_comments`. Each post is assigned one of four engagement archetypes using per-subreddit 50th percentile thresholds on both columns:

| Label | Score | Comments |
|---|---|---|
| `viral` | ≥ median | ≥ median |
| `crowd-pleaser` | ≥ median | < median |
| `debate-starter` | < median | ≥ median |
| `low-engagement` | < median | < median |

Per-subreddit thresholds are used rather than global thresholds because engagement norms vary drastically across communities — 100 comments is unremarkable in r/AskReddit but exceptional in a small niche subreddit. We expect significant class imbalance toward `low-engagement`, which will be addressed via inverse-frequency class weights during training.

### Do you have missing and duplicate values in your dataset?

#### Null Values

Null counts are low across almost all columns:

| Column | Null Count | % Missing |
|---|---|---|
| `subreddit` / `subreddit_id` | 306,479 | 0.056% |
| `score` | 21 | ~0% |
| All others | 0 | 0% |

#### Sentinel Strings

A critical data quality issue is Pushshift's use of sentinel strings instead of SQL NULLs and a naive null check will miss these entirely. Of the 549M `selftext` values, **399.5M are empty strings** (link posts with no body) and **150.1M contain actual text**. There are no NULL, `[removed]`, or `[deleted]` values in `selftext` in this version of the dataset and all placeholder content was stored as empty strings.

For `author`, deleted and anonymous accounts are encoded as `[deleted]` and empty strings rather than NULLs:

| Author Type | Count | % of Total |
|---|---|---|
| Real users | 444,817,691 | 80.9% |
| `[deleted]` | 73,272,560 | 13.3% |
| Empty string | 30,256,874 | 5.5% |
| `AutoModerator` | 1,315,830 | 0.2% |

#### Negative `num_comments` Values

`num_comments` has a minimum value of -117, which is a known Pushshift artifact where comment counts were recorded before Reddit's counters fully settled. These are invalid records rather than true missing values.

#### Duplicate Posts

Only **1 duplicate post ID** was found across 549,662,955 rows.

### Key Visualizations

#### Post Score Distribution

![Score Distribution](outputs/figures/plot3_score_distribution.png)

The score distribution is heavily right-skewed across all 549M posts. The vast majority of posts score in the 1–9 range, with each successive order of magnitude containing dramatically fewer posts. This confirms that raw score alone is a poor engagement signal. Collapsing this distribution into a binary high/low label would classify nearly everything as low-engagement. A per-subreddit quantile-based archetype label is a more meaningful and discriminative target.

#### Score vs. Comment Count (Log-Log Scale, n=50k Sample)

![Score vs Comments](outputs/figures/plot4_score_vs_comments.png)

A 50,000-post sample plotted on a log-log scale shows a positive but noisy relationship between score and comment count. While the OLS trend line confirms a weak positive correlation, the substantial variance around it demonstrates that the two metrics capture meaningfully different engagement phenomena. Posts in the upper-left region (high comments, low score) represent debate-starters i.e., controversial content that drives discussion without broad approval. Posts in the lower-right (high score, low comments) represent crowd-pleasers, broadly liked but not discussion-generating. This scatter is the core visual justification for a four-archetype label over a single engagement metric.

## Preprocessing Plan

Preprocessing will be implemented in Milestone 3 using PySpark DataFrame operations. A key consideration is Pushshift's use of empty strings instead of SQL NULLs for missing `selftext`. A naive null check will miss these entirely. Bot and deleted author accounts are also prevalent and must be handled explicitly.

### Missing Values
- `title`: Rows with zero-length titles will be dropped as they contain no usable signal.
- `selftext`: Empty strings indicate link posts and are valid they will be flagged via a binary `is_text_post` indicator column rather than treated as missing.
- `author`: Rows with `[deleted]` or empty string authors will be filtered out as they cannot contribute to author history features. Authors posting at an inhuman frequency (above a defined daily post threshold) will be flagged as suspected bots and considered for removal.
- `subreddit` / `subreddit_id`: The ~306K null rows discovered in EDA will be dropped as they cannot contribute to per-subreddit features.
- `score` / `num_comments`: The 21 null score rows will be dropped. Negative `num_comments` values (-117 minimum) are a known Pushshift artifact and will be clipped to 0.

### Duplicate Posts
Only 1 duplicate post ID was found in EDA. All duplicate IDs will be deduplicated as a precaution.

### Class Imbalance
We expect the majority of posts to fall into the low-engagement class. This will be addressed using PySpark MLlib's `weightCol` parameter to assign inverse-frequency class weights during training, giving minority classes (viral, crowd-pleaser, debate-starter) proportionally higher influence on the loss function.

### Label Generation
Labels will be assigned using `pyspark.sql.Window` functions to compute per-subreddit 50th percentile thresholds for `score` and `num_comments`, then applying a four-way conditional to assign each post its engagement archetype. A two-tier high vs. low engagement label will also be generated for comparison against the four-class model.

### Feature Engineering
All features will be derived from pre-publication information only to prevent data leakage. Transformations include:

- **Title features**: character count, word count, question/exclamation presence, sentiment polarity, readability extracted using PySpark UDFs wrapping NLTK/TextBlob
- **Post type**: binary `is_text_post` flag derived from `selftext`; selftext character length for text posts
- **Author history**: mean score, posting frequency, subreddit diversity computed via `groupBy().agg()` aggregations across the full dataset
- **Subreddit context**: post volume, median score, median comment count computed via `groupBy().agg()` per subreddit
- **Temporal**: hour of day and day of week extracted from `created_utc` via `pyspark.sql.functions.from_unixtime`

### Scaling and Encoding
- All continuous features will be standardized using MLlib's `StandardScaler`
- Categorical features (`subreddit`, `author`) will not be directly one-hot encoded due to high cardinality (millions of unique authors, hundreds of thousands of subreddits). Each is represented through aggregated numerical features author history features for `author` and community context features for `subreddit`
- The final feature vector will be assembled using MLlib's `VectorAssembler`

### Key PySpark Operations
- `df.cache()` — prevent repeated full dataset scans across multiple actions
- `df.filter()` — remove null and invalid rows
- `df.fillna()` — impute missing values where applicable
- `Window.partitionBy()` + `percentile_approx()` — per-subreddit quantile thresholds for label generation
- `df.groupBy().agg()` — author and subreddit feature aggregations
- `df.withColumn()` + UDFs — title linguistic feature extraction
- `VectorAssembler` — final feature vector construction
- `StandardScaler` — feature normalization

## Milestone 3 — Preprocessing and XGBoost Model Fitting (4-Class and Binary)

### Notebook (Milestone 3)

The Milestone 3 preprocessing and model fitting notebook is located at: [`notebooks/Milestone3_Pushshift.ipynb`](notebooks/Milestone3_Pushshift.ipynb)

### SparkSession Configuration (Milestone 3)

Driver memory was increased to 120GB to accommodate the full dataset pipeline. All execution runs in `local[*]` mode with 16 cores as parallel thread slots within a single JVM. Spark local spill directory was redirected to Lustre to account for the expensive computations that come with processing 535M rows using heavy aggregations.

```python
spark = SparkSession.builder \
    .appName("PushshiftRedditPreprocessing") \
    .config("spark.driver.memory", "120g") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.parquet.enableVectorizedReader", "true") \
    .config("spark.local.dir", "/expanse/lustre/projects/uci157/ekim18/spark-tmp") \
    .getOrCreate()
```

### SparkUI Screenshot (Milestone 3)

Training used 16 Spark workers with `SparkXGBClassifier`, confirmed via the Spark REST API executor summary:

<img width="1012" height="401" alt="image" src="https://github.com/user-attachments/assets/dd406292-4d30-4a1f-b413-b7de4a16f6e8" />

### Preprocessing Pipeline (Milestone 3)

All preprocessing was implemented using Spark DataFrame operations and Spark MLlib transformers on the full 535,480,818 row filtered dataset on SDSC Expanse.

**Filtering and Cleaning:**

| Filter | Rows Removed | Reason |
|---|---|---|
| Null `subreddit` / `subreddit_id` | ~306,479 | Cannot contribute to per-subreddit features |
| Null `score` | 21 | Required for label generation |
| Negative `num_comments` | 1,090 | Pushshift artifact, value unverifiable |
| Duplicate `id` | 0 | Only 1 duplicate row existed and removal was not worth the compute |

**Label Generation:**
Per-subreddit 75th percentile thresholds on `score` and `num_comments` were computed using `Window.partitionBy()` and `percentile_approx()` to assign four engagement archetypes. A binary high/low label was also generated for comparison. StringIndexer was applied to both label columns.

| Label | Class | Distribution |
|---|---|---|
| Low-engagement | 0 | 44.3% |
| Viral | 1 | 29.9% |
| Crowd-pleaser | 2 | 14.0% |
| Debate-starter | 3 | 11.8% |

**Binary Label Distribution:**

| Label | Class | Distribution |
|---|---|---|
| Low-engagement | 0 | 44.3% |
| High-engagement | 1 | 55.7% |

**Features (19 total, all pre-publication):**
- Title structural: `title_len`, `has_question`, `has_exclamation`, `title_has_number`, `title_is_allcaps`
- Post type: `has_title`, `is_text_post`, `selftext_len`
- Author flags: `is_known_bot`, `is_anonymous_author`
- Temporal: `hour_of_day`, `day_of_week`
- Author history aggregations: `author_post_count`, `author_mean_score`
- Subreddit context aggregations: `subreddit_post_count`, `subreddit_median_score`, `subreddit_median_comments`
- NLP sentiment (VADER UDF): `title_sentiment`, `selftext_sentiment`

**MLlib Pipeline:** Imputer (mean, `selftext_sentiment` only) → StringIndexer → VectorAssembler → StandardScaler (fit on train only)

**Train/Val/Test Split (temporal):**

| Split | Years | Rows |
|---|---|---|
| Train | 2012–2016 | 276,442,594 |
| Val | 2017 | 114,089,205 |
| Test | 2018 | 144,949,019 |

### Model Training (Milestone 3)

We trained two XGBoost configurations (`SparkXGBClassifier`, XGBoost 2.0.3) on both the 4-class and binary label tasks so four models total. All models were trained on the full 276M row training split with 16 workers and 16 partitions (~17M rows per worker).

**Config A — Conservative (shallow, regularized):**

| Parameter | Value |
|---|---|
| `max_depth` | 4 |
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 5.0 |
| `reg_alpha` | 1.0 |
| `min_child_weight` | 50 |

**Config B — Aggressive (deep, fast learning):**

| Parameter | Value |
|---|---|
| `max_depth` | 8 |
| `n_estimators` | 300 |
| `learning_rate` | 0.1 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.7 |
| `reg_lambda` | 1.0 |
| `reg_alpha` | 0.0 |
| `min_child_weight` | 10 |

**Training Times:**

| Model | Config A | Config B |
|---|---|---|
| 4-Class | 3,610.9s (~1hr) | 7,059.0s (~2hrs) |
| Binary | ~3,600s (~1hr) | ~7,200s (~2hrs) |

### Model Evaluation Results (Milestone 3)

**4-Class Weighted F1:**

| Config | Train | Val | Test |
|---|---|---|---|
| Config A | 0.5283 | 0.5759 | 0.5689 |
| Config B | 0.5767 | 0.6089 | 0.5866 |

**Binary Weighted F1:**

| Config | Train | Val | Test |
|---|---|---|---|
| Config A | 0.7087 | 0.7435 | 0.7280 |
| Config B | 0.7375 | 0.7642 | 0.7440 |

**4-Class vs Binary Comparison (Test Weighted F1):**

| Config | 4-Class WF1 | Binary WF1 | Delta |
|---|---|---|---|
| Config A | 0.5689 | 0.7280 | +0.1590 |
| Config B | 0.5866 | 0.7440 | +0.1574 |

### Model Fitting Analysis (Milestone 3)

![Fitting Analysis](outputs/figures/xgboost_fitting_analysis.png)

Both models sit firmly in the underfitting region of the bias-variance spectrum. Train F1 is consistently lower than val and test F1 across all configurations and both label tasks which is the opposite of the overfitting pattern. This is probably driven by a temporal shift: the 2012–2016 training set covers Reddit's noisy early growth phase, while the 2017 val and 2018 test sets represent a more behaviorally stable platform where posting patterns are more consistent and predictable.

Config B outperforms Config A across every split and both tasks. The deeper trees and faster learning rate allow Config B to capture more complex feature interactions, particularly the relationship between subreddit-level aggregates and engagement thresholds.

### Feature Importance (Milestone 3)

![4-Class Feature Importance](outputs/figures/xgboost_4class_feature_importance.png)

![Binary Feature Importance](outputs/figures/xgboost_binary_feature_importance.png)

Community and author context features dominate both tasks with our current feature set. `subreddit_post_count`, `author_mean_score`, and `author_post_count` are the top three features in every configuration. This implies knowing which community a post belongs to and the historical track record of its author is far more predictive than any structural property of the post itself. `has_question` and `has_exclamation` contribute zero weight in Config A across both label tasks and are candidates for removal in Milestone 4.

### Speedup Analysis (Milestone 3)

The VADER selftext sentiment UDF applied to 535,480,818 rows was used as the representative distributed operation for speedup measurement.

| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 25,914 (estimated) | 1.00x | 100% |
| 16 | 2,308 | 11.23x | 70.2% |

**Speedup** = T₁ / T₁₆ = 25,914 / 2,308 = **11.23x**

**Efficiency** = 11.23 / 16 = **70.2%**

Using Amdahl's Law, the implied parallel fraction is approximately 96.4%, with the remaining ~3.6% serial fraction corresponding to task scheduling overhead, Python UDF initialization, and VADER analyzer instantiation per partition. The single-thread baseline is estimated from a 10,000-row benchmark sample extrapolated to the full dataset. Running the full pipeline single-threaded would require approximately 7 hours of wall time beyond the project's time constraints. 

### Conclusion (Milestone 3)

The 4-class XGBoost models demonstrate that Reddit engagement archetypes are meaningfully predictable from pre-publication features, achieving a test weighted F1 of 0.5866 (Config B) on a 4-class problem across 102,739 subreddits using only 19 features. Both models underfit due to a hard ceiling imposed by the pre-publication feature set. Without access to post content, the model cannot distinguish crowd-pleaser from debate-starter posts reliably, as reflected in the low per-class F1 scores for these minority classes (0.33–0.38 on test). The binary task confirms that the high/low engagement distinction is substantially more learnable from these features, achieving 0.7440 test WF1, showing consistent ~0.16 point lift over the 4-class task across all configurations.

**Improvements for Milestone 4:** Incorporating Word2Vec title embeddings to add semantic content signal, removing zero-weight features (`has_question`, `has_exclamation`), and potentially exploring LightGBM as an alternative to XGBoost given its faster training on large datasets. A neural net model could be explored too. 

**How distributed computing helped:** The preprocessing pipeline involved per-author and per-subreddit aggregations, VADER sentiment UDF, and StandardScaler fit, all on 535M rows and exceeds single-machine memory capacity. Spark's distributed execution achieved an 11.23x speedup over single-thread baseline. Model training itself used 16 parallel XGBoost workers via `SparkXGBClassifier`, reducing training time to ~1–2 hours per model compared to an estimated 16+ hours single-threaded.



## Milestone 4 — Dimensionality Reduction (SVD/LSA) and Logistic Regression


### Overview

Milestone 3 established that test performance is structurally bounded: roughly 48.6% of the 2018 test set consists of authors with no training-period history, so the author/subreddit aggregate features that dominate the baseline carry no signal for nearly half of test. Content-derived features are then our focus for Milestone 4.

This milestone reduces the sparse TF-IDF representation using **Truncated SVD**, the standard form of **Latent Semantic Analysis (LSA)** for TF-IDF, then trains logistic regression models on the resulting combined feature vector.

### SparkSession Configuration (Milestone 4)

```python
spark = SparkSession.builder \
    .appName("PushshiftRedditSVD") \
    .master("local[*]") \
    .config("spark.driver.memory", "120g") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.parquet.enableVectorizedReader", "true") \
    .config("spark.local.dir", "/expanse/lustre/projects/uci157/ekim18/spark-tmp") \
    .getOrCreate()
```

### TF-IDF Feature Engineering

A 10,000-dimensional TF-IDF representation of post titles was constructed using `pyspark.ml.feature.HashingTF` + `IDF`, fit on the training split only and applied to all three splits.

### Truncated SVD (Latent Semantic Analysis)

`pyspark.mllib.linalg.distributed.RowMatrix.computeSVD` was used rather than `pyspark.ml.feature.PCA` deliberately: PCA mean-centers the matrix, which turns every structural zero into a nonzero entry and is much tougher on memory. LSA preserves sparsity.

```python
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.linalg.distributed import RowMatrix

K = 100
N_HASH_FEATURES = 10000

# Convert ml.linalg.SparseVector -> mllib.linalg.Vector (sparsity-preserving bridge)
train_rows = (
    train_sample
    .select("title_tfidf")
    .rdd
    .map(lambda r: MLLibVectors.fromML(r.title_tfidf))
).persist(StorageLevel.MEMORY_AND_DISK)

mat = RowMatrix(train_rows, numRows=n_train, numCols=N_HASH_FEATURES)

# computeU=False: only V (10000 x k, local) and s needed for projection
svd = mat.computeSVD(K, computeU=False)
```

The SVD was fit on a ~2M-row subsample of the training split. The full 83M-row fit was extrapolated at ~33 , not viable before submission, so the subsample fit serves as the production SVD: the leading singular directions of a 10k TF-IDF space stabilize well below the full row count. `V` (10,000 × 100) and `s` (100 singular values) were persisted to `../models/svd_title_k100_v2/`.

**Explained energy results:** The singular values are nearly flat after the first component (σ₁ = 1,244; components 2–100 ≈ 550–880). Cumulative explained energy reaches 7.9% at k=100, rising almost linearly across the retained range with no elbow.

![SVD Explained Energy](outputs/figures/svd_explained_energy.png)

### SVD Transform

Each split's `title_tfidf` vector was projected into the 100-dimensional SVD subspace via a broadcast UDF operating only over nonzero indices:

```python
V_bc = spark.sparkContext.broadcast(svd.V.toArray().astype(np.float32))  # 10000 x 100

@F.udf(VectorUDT())
def project(v):
    out = np.zeros(100, dtype=np.float32)
    Vmat = V_bc.value
    for idx, val in zip(v.indices, v.values):
        out += val * Vmat[idx]
    return Vectors.dense(out)

for name, df in {"train": train_tfidf, "val": val_tfidf, "test": test_tfidf}.items():
    df.withColumn("svd_features", project("title_tfidf")) \
      .drop("title_tfidf") \
      .write.mode("overwrite").parquet(OUT[name])
```

### Feature Assembly (117-dim vector)

Logistic regression cannot handle the NaN cold-start aggregates that XGBoost managed natively. The five nullable aggregate features were median-imputed (Imputer fit on train only); `is_new_author` / `is_new_subreddit` flags mark imputed rows. Four zero-importance structural features (`has_title`, `has_question`, `has_exclamation`, `title_is_allcaps`) identified in M3 were dropped.

Final vector: **12 row-local structured + 5 imputed aggregates + 100 SVD components = 117 dimensions**

```python
from pyspark.ml.feature import Imputer, VectorAssembler

AGG    = ["author_post_count","author_mean_score","subreddit_post_count",
          "subreddit_median_score","subreddit_median_comments"]
NONAGG = ["title_len","title_has_number","is_text_post","selftext_len",
          "hour_of_day","day_of_week","is_known_bot","is_anonymous_author",
          "is_new_author","is_new_subreddit","title_sentiment","selftext_sentiment"]

imputer = Imputer(strategy="median", inputCols=AGG,
                  outputCols=[c+"_imp" for c in AGG]).fit(train_s)

assembler = VectorAssembler(
    inputCols=NONAGG + [c+"_imp" for c in AGG] + ["svd_features"],
    outputCol="features", handleInvalid="error")
```

### Logistic Regression Training

Four logistic regression models were trained: multinomial (4-class) and binomial (binary), each in a conservative (A, regParam=0.1) and lighter (B, regParam=0.01) L2 regularization configuration, mirroring M3's Config A/B structure.

```python
from pyspark.ml.classification import LogisticRegression

specs = [
    ("lr_4class_A", "label_4class_idx", "weight_4class", "multinomial", 0.1),
    ("lr_4class_B", "label_4class_idx", "weight_4class", "multinomial", 0.01),
    ("lr_binary_A", "label_binary_idx", "weight_binary", "binomial",    0.1),
    ("lr_binary_B", "label_binary_idx", "weight_binary", "binomial",    0.01),
]
for key, lab, wt, fam, rp in specs:
    lr = LogisticRegression(featuresCol="features", labelCol=lab, weightCol=wt,
                            family=fam, maxIter=100, regParam=rp, elasticNetParam=0.0)
    m = lr.fit(train_a)
    m.write().overwrite().save(f"../models/{key}_v2/")
```

### Model Evaluation Results (Milestone 4)

**4-Class and Binary Weighted F1:**

| Model | regParam | Train WF1 | Val WF1 | Test WF1 |
|---|---|---|---|---|
| lr_4class_A | 0.10 | 0.4324 | 0.4783 | 0.4707 |
| lr_4class_B | 0.01 | 0.4424 | 0.4913 | 0.4793 |
| lr_binary_A | 0.10 | 0.6370 | 0.6744 | 0.6683 |
| lr_binary_B | 0.01 | 0.6464 | 0.6815 | 0.6723 |

*M3 XGBoost baselines for reference: 4-class test WF1 = 0.4514 (Config A); binary test WF1 = 0.6260 (Config A).*

The lighter-regularization Config B models edge out Config A on every split. Both 4-class (0.4793) and binary (0.6723) test WF1 exceed the structured-only M3 XGBoost baselines, confirming that SVD content features add predictive signal beyond the structured feature set alone.

**WF1 by Split:**

![LR WF1 by Split](outputs/figures/m4_lr_wf1_by_split.png)

**Confusion Matrices (Config B, test split):**

![4-Class Confusion Matrix](outputs/figures/m4_lr_4class_B_confusion_test.png)

![Binary Confusion Matrix](outputs/figures/m4_lr_binary_B_confusion_test.png)

Test WF1 exceeds train WF1 across all four models. This is consistent with the evaluation design rather than anomalous generalization: training predictions come from class-weighted fits over the balanced 30% stratified sample, while val/test carry the natural class distribution. The near-zero train/test gap combined with low minority-class F1 (crowd-pleaser ~0.19, debate-starter ~0.23) indicates high bias in the linear model it lacks the capacity to separate the middle engagement tiers, not the variance that would indicate overfitting.


---

# Written Report

## Abstract

Social media platforms face a critical challenge in predicting how content will engage their communities. Not just whether a post will be popular, but what kind of engagement it will receive. A post that attracts thousands of upvotes but yields no comment discussion serves a fundamentally different role than one that sparks hundreds of comments but remains controversial in score (whether it be likes, reddit votes, etc.), yet most engagement prediction approaches collapse these distinct outcomes into a single metric. This project uses the Pushshift Reddit Submissions Dataset, a public archive of Reddit posts hosted on HuggingFace, totaling approximately 89 GB and containing over 549 million submissions. We propose a multiclass classification pipeline in PySpark that predicts a post's engagement archetype (viral, crowd-pleaser, debate-starter, or low-engagement) using pre-publication features derived from title linguistics, author posting history, subreddit context, and temporal patterns. Class boundaries are defined using per-subreddit quantiles of score and comment count to account for the vastly different engagement norms across communities. This analysis requires distributed processing because constructing per-author behavioral features across hundreds of millions of posts, computing per-subreddit engagement baselines, and performing temporal aggregations exceed the memory and compute capacity of a single machine.

## Introduction

Reddit is one of the world’s largest social media platforms, organized into thousands of distinct communities known as subreddits. Each subreddit has its own norms, audience, and engagement patterns. Understanding the factors that drive post engagement can support content moderation, platform health, and proactive resource allocation, especially for identifying posts likely to generate contentious or high-volume discussion.

Existing work on social media engagement prediction typically frames the problem as binary (popular vs. not) or as a regression over raw vote counts. This conflates qualitatively different engagement outcomes: a post can accumulate high scores with minimal discussion, or generate extensive debate while remaining score-neutral or negative. We argue that a further segmented engagement archetype derived from the joint distribution of score and comment volume relative to community norms is a more meaningful and actionable prediction target than a simple high vs. low engagement model. Our project seeks to gain further classification insight from the juxtaposition of our proposed four-tier class model to a standard two-tier class model.

We define four engagement archetypes:
- **Viral**: high score, high comments — broadly appealing and discussion-generating
- **Crowd-pleaser**: high score, low comments — well-received but not discussion-driving
- **Debate-starter**: low score, high comments — controversial or polarizing content
- **Low-engagement**: low score, low comments — did not resonate with the community

Class boundaries are assigned using per-subreddit quantile thresholds rather than global thresholds, accounting for the fact 
that 100 comments is unremarkable in r/AskReddit but exceptional in a small niche subreddit. 


## Methods

### Data Exploration

The full 549,662,955-post dataset was profiled in Milestone 2 using PySpark DataFrame operations on SDSC Expanse. Key findings: both `score` (mean 44.84, std 707.38) and `num_comments` (mean 8.36, std 93.11) are extremely right-skewed, confirming that global engagement thresholds would label nearly everything as low-engagement. This led us to believe that looking locally at subreddits and evaluating engagement on quantiles for each subreddit would be more effective.

Additionally, a critical data quality issue was discovered: Pushshift stores missing `selftext` as empty strings rather than SQL NULLs, and deleted authors as the literal string `[deleted]`. Naive null-checks miss both entirely.

### Preprocessing

All preprocessing ran on the full 535,480,818-row filtered dataset in PySpark on SDSC Expanse.

**Filtering:** 
Rows with null `subreddit`/`subreddit_id` (~306K), null `score` (21), and negative `num_comments` (1,090) were removed. The single duplicate post ID was retained as the removal was not worth the compute cost.

**Label generation:** 
Per-subreddit 75th percentile thresholds for `score` and `num_comments` were computed via `Window.partitionBy("subreddit")` + `percentile_approx()`, then a four-way conditional assigned each post its engagement archetype. A binary high/low label was generated in parallel.

**Feature engineering (19 features, all pre-publication):** 
Title structural features (length, number presence, sentiment via VADER UDF), post-type flags (`is_text_post`, `selftext_len`), author flags (`is_known_bot`, `is_anonymous_author`), temporal features (`hour_of_day`, `day_of_week`), author history aggregations (`author_post_count`, `author_mean_score`), and subreddit context aggregations (`subreddit_post_count`, `subreddit_median_score`, `subreddit_median_comments`).

**Train/val/test split:** 
A stratified split with train (276M rows), validation (114M rows), and test (145M rows).


### Model 1: Distributed XGBoost on Structured Features

`SparkXGBClassifier` (XGBoost 2.0.3) was trained on the full 276M-row training split. Two configurations were trained on both the 4-class and binary tasks (four models total):

**Config A (conservative):** `max_depth=4`, `n_estimators=200`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5.0`, `reg_alpha=1.0`, `min_child_weight=50`

**Config B (aggressive):** `max_depth=8`, `n_estimators=300`, `learning_rate=0.1`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_lambda=1.0`, `reg_alpha=0.0`, `min_child_weight=10`

Inverse-frequency class weights were applied via `weightCol`. Training took ~1 hour (Config A) and ~2 hours (Config B) per task.

Notably, the training data here was only our structural data and did not include any TF-IDF or other NLP methods.


### Final Model: Truncated SVD (LSA) + Logistic Regression

**TF-IDF:** A 10,000-dimensional TF-IDF representation of post titles was constructed using `HashingTF` + `IDF`, fit on training only and applied to all three splits.

**Truncated SVD:** `RowMatrix.computeSVD(k=100, computeU=False)` was applied to the training-split TF-IDF matrix. The quantity reported is explained Frobenius energy (uncentered), not centered statistical variance. The SVD was fit on a ~2M-row training subsample; the leading singular directions of a 10k TF-IDF space stabilize well below full row count.

**Feature assembly (117-dim):** The 100 SVD components were concatenated with 17 retained structured features (the 21-feature M3 vector minus `has_title`, `has_question`, `has_exclamation`, `title_is_allcaps`, which had zero importance in M3). Nullable features were median-imputed (Imputer fit on train only).

**Logistic regression:** Four models multinomial (4-class) and binomial (binary), each in regParam=0.1 (Config A) and regParam=0.01 (Config B) L2 regularization, were trained on a 30% stratified sample with inverse-frequency class weights. Evaluation used one distributed `groupBy(label, prediction)` per split to compute confusion matrices with minimum full-data passes.


## Results


### Model 1 Results: XGBoost on Structured Features

**4-Class Weighted F1:**

| Config | Train | Val | Test |
|---|---|---|---|
| Config A | 0.5283 | 0.5759 | 0.5689 |
| Config B | 0.5767 | 0.6089 | 0.5866 |

**Binary Weighted F1:**

| Config | Train | Val | Test |
|---|---|---|---|
| Config A | 0.7087 | 0.7435 | 0.7280 |
| Config B | 0.7375 | 0.7642 | 0.7440 |

Our first approach consistently had lower train F1 score than the validation and test splits. Feature importance is dominated by `subreddit_post_count`, `author_mean_score`, and `author_post_count`; `has_question` and `has_exclamation` contribute zero weight in Config A.

![XGBoost Fitting Analysis](outputs/figures/xgboost_fitting_analysis.png)

![4-Class Feature Importance](outputs/figures/xgboost_4class_feature_importance.png)

![Binary Feature Importance](outputs/figures/xgboost_binary_feature_importance.png)



### Final Model Results: SVD + Logistic Regression

**Explained Frobenius energy:**

| k | Cumulative Explained Energy |
|---|---|
| 1 | 0.29% |
| 10 | 1.40% |
| 50 | 4.80% |
| 100 | 7.90% |

The spectrum is nearly flat after the first component (σ₁ = 1,244; components 2–100 range ~878 to ~548) with no elbow, indicating high effective rank. k=100 was chosen as a pragmatic compute/performance tradeoff.

![SVD Explained Energy](outputs/figures/svd_explained_energy.png)

**Logistic Regression Weighted F1:**

| Model | regParam | Train WF1 | Val WF1 | Test WF1 |
|---|---|---|---|---|
| lr_4class_A | 0.10 | 0.4324 | 0.4783 | 0.4707 |
| lr_4class_B | 0.01 | 0.4424 | 0.4913 | 0.4793 |
| lr_binary_A | 0.10 | 0.6370 | 0.6744 | 0.6683 |
| lr_binary_B | 0.01 | 0.6464 | 0.6815 | 0.6723 |

Both 4-class (0.4793) and binary (0.6723) test WF1 exceeded what we could achieve with just the structural data, confirming that SVD title content features add signal beyond the structured feature set.

![LR WF1 by Split](outputs/figures/m4_lr_wf1_by_split.png)

**Confusion matrices (Config B, test):**

![4-Class Confusion Matrix](outputs/figures/m4_lr_4class_B_confusion_test.png)

![Binary Confusion Matrix](outputs/figures/m4_lr_binary_B_confusion_test.png)

Per-class breakdown for lr_4class_B on the test split: low-engagement is the best-classified class (~0.65 recall). Crowd-pleaser and debate-starter show the lowest F1 scores (~0.19 and ~0.23 respectively), frequently misclassified into low-engagement.



## Discussion

**Lack of power of first model**
The first model we created was lacking in F1 score, largely due to the lack of signal from the text data. We attempted to do predictions based off of the meta-data from the posts as well as a quick sentiment score. The poor performance of the first model led us to our final model incorporating TF-IDF.

**Explained energy vs. predictive signal.** 
The SVD retains only 7.9% of total Frobenius energy at k=100. This initially seems discouraging, but the downstream logistic regression results demonstrate that retained energy and retained predictive signal are distinct quantities: the 100-component representation, concatenated with structured features, improves over the structured-only baseline on both tasks. Further research could be done to see the lift of additional dimensionality for the SVD step.

**Fitting regime.** 
The model sits on the underfitting side for the 4-class task and near balanced for binary. The results indicate high bias as the linear decision boundaries cannot separate the middle engagement tiers. Lighter regularization (Config B) improves all splits over Config A, which indicated the model benefits from more flexibility. Additional depth in model configuration may allow for better fits for the 4-class task.

**Component interpretability.** Because TF-IDF was computed with `HashingTF` (a one-way hash with no inverse), there is no recoverable mapping from SVD component loadings back to vocabulary terms. Individual component interpretation is not possible with the current artifacts. A future iteration using `CountVectorizer` would enable this at the cost of holding the full vocabulary in memory.


## Conclusion

This project demonstrates that Reddit engagement archetypes are meaningfully predictable from pre-publication features, especially at the binary (high engagement vs low engagement) level. At further granularity it becomes obvious that richer context is required for effective modeling. We learned that distributed computing allowed us to work with data that would have otherwise been impossible. The ability to do analysis across hundreds of millions of rows of data would be highly inefficient or impossible on a single machine. The caveat to this is that the Spark framework was prone to out of memory errors and kernel failures. So we often had to revisit our approaches to better fit the distributed system, focusing on delegating processes into smaller pieces that wouldn't overwhelm the distributed system. With more time and resources we would attempt replacing HashingTF with CountVectorizer which would allow for greater interpretability. Additionally, we'd want to increase the SVD rank to k = 500 or k = 1000 on a true multi-node cluster rather than a local model.



## Contributions

| Member | Contributions |
|---|---|
| Eugene Kim | *To be completed* |
| Jack Keeton | Led development for data ingestion and initial exploration (Milestone 2). Aided efforts in feature engineering and tested the application of word embeddings. Developed GBT Classifier alternative for final model.|
| Aidan Sanchez | *To be completed* |

## References

- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit Dataset. *Proceedings of the International AAAI Conference on Web and Social Media*, 14, 830–839.
- Apache Spark MLlib Documentation: https://spark.apache.org/docs/latest/ml-guide.html
- HuggingFace Datasets: https://huggingface.co/datasets/fddemarco/pushshift-reddit
