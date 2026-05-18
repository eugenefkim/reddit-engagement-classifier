# reddit-engagement-classifier

Distributed PySpark ML pipeline that classifies Reddit post engagement archetypes (viral, crowd-pleaser, debate-starter, low-engagement) using NLP feature engineering on 89GB of Pushshift Reddit data.

## Abstract

Social media platforms face a critical challenge in predicting how content will engage their communities. Not just whether a post will be popular, but what kind of engagement it will receive. A post that attracts thousands of upvotes but yields no comment discussion serves a fundamentally different role than one that sparks hundreds of comments but remains controversial in score (whether it be likes, reddit votes, etc.), yet most engagement prediction approaches collapse these distinct outcomes into a single metric. This project uses the Pushshift Reddit Submissions Dataset, a public archive of Reddit posts hosted on HuggingFace, totaling approximately 89 GB and containing over 549 million submissions. We propose a multiclass classification pipeline in PySpark that predicts a post's engagement archetype (viral, crowd-pleaser, debate-starter, or low-engagement) using pre-publication features derived from title linguistics, author posting history, subreddit context, and temporal patterns. Class boundaries are defined using per-subreddit quantiles of score and comment count to account for the vastly different engagement norms across communities. This analysis requires distributed processing because constructing per-author behavioral features across hundreds of millions of posts, computing per-subreddit engagement baselines, and performing temporal aggregations exceed the memory and compute capacity of a single machine.

## Introduction

Reddit is one of the largest social platforms in the world, organized into thousands of distinct communities (subreddits) each with their own norms, audiences, and engagement patterns. Understanding what drives post engagement has practical implications for content moderation, platform health, and proactive resource allocation, particularly for identifying posts likely to generate contentious or high-volume discussion before comments arrive.

Existing work on social media engagement prediction typically frames the problem as binary (popular vs. not) or as a regression over raw vote counts. This conflates qualitatively different engagement outcomes: a post can accumulate high scores with minimal discussion, or generate extensive debate while remaining score-neutral or negative. We argue that a further segmented engagement archetype derived from the joint distribution of score and comment volume relative to community norms is a more meaningful and actionable prediction target than a simple high vs. low engagement model. Our project seeks to gain further classification insight from the juxtaposition of our proposed four-tier class model to a standard two-tier class model.

We define four engagement archetypes:
- **Viral**: high score, high comments — broadly appealing and discussion-generating
- **Crowd-pleaser**: high score, low comments — well-received but not discussion-driving
- **Debate-starter**: low score, high comments — controversial or polarizing content
- **Low-engagement**: low score, low comments — did not resonate with the community

Class boundaries are assigned using per-subreddit quantile thresholds rather than global thresholds, accounting for the fact that 100 comments is unremarkable in r/AskReddit but exceptional in a small niche subreddit. 

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
- **`author`** (string): Reddit username of the poster. ~35.4M distinct users. Heavily skewed — the top authors are bots and aggregator accounts (`AutoNewsAdmin`, `AutoNewspaperAdmin`, `politicbot`, etc.) responsible for millions of posts each. ~13.3% of posts have `[deleted]` authors and ~5.5% have empty string authors. Not directly encoded as a feature; represented through author history aggregations.
- **`subreddit`** (string): Community the post was submitted to. ~2.3M distinct subreddits. Extremely long-tailed — r/AskReddit alone accounts for ~13.5M posts while the vast majority of subreddits have very few. Not directly encoded; represented through per-subreddit aggregation features.
- **`subreddit_id`** (string): Internal Reddit identifier for the subreddit. ~2.2M distinct values, closely tracking `subreddit` cardinality. Used for deduplication and join purposes only.
- **`selftext`** (string): Post body text. 72.8% of posts have empty selftext, indicating link or image posts with no body. The remaining 27.2% are text posts. No null or sentinel string values present in this version of the dataset — absence is encoded as an empty string. Represented as a binary `is_text_post` flag and selftext character length in modeling.
- **`title`** (string): Post title text. Present for all 549M rows. Mean length ~50 characters, std dev ~37, max 1,592 characters. Used as the primary source of NLP features (word count, sentiment, readability, question/exclamation presence).

#### Continuous Columns

- **`score`** (long): Net upvotes at time of archival. Range: 0 to 270,469. Mean: 44.84, std dev: 707.38 — extremely right-skewed. The vast majority of posts score in the single digits while a small number go viral. Median is consistently far below mean in every year. One of the two **target label inputs**.
- **`num_comments`** (long): Comment count at time of archival. Range: -117 to 517,003. Mean: 8.36, std dev: 93.11 — similarly right-skewed. Negative values are a known Pushshift artifact and will be treated as 0. The other **target label input**.
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
- `selftext`: Empty strings indicate link posts and are valid — they will be flagged via a binary `is_text_post` indicator column rather than treated as missing.
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

- **Title features**: character count, word count, question/exclamation presence, sentiment polarity, readability — extracted using PySpark UDFs wrapping NLTK/TextBlob
- **Post type**: binary `is_text_post` flag derived from `selftext`; selftext character length for text posts
- **Author history**: mean score, posting frequency, subreddit diversity — computed via `groupBy().agg()` aggregations across the full dataset
- **Subreddit context**: post volume, median score, median comment count — computed via `groupBy().agg()` per subreddit
- **Temporal**: hour of day and day of week extracted from `created_utc` via `pyspark.sql.functions.from_unixtime`

### Scaling and Encoding
- All continuous features will be standardized using MLlib's `StandardScaler`
- Categorical features (`subreddit`, `author`) will not be directly one-hot encoded due to high cardinality (millions of unique authors, hundreds of thousands of subreddits). Each is represented through aggregated numerical features — author history features for `author` and community context features for `subreddit`
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

| id | totalCores | maxMemory | activeTasks | isActive | maxMemory_GB |
|---|---|---|---|---|---|
| driver | 16 | 77,120,667,648 | 0 | True | 71.82 |

Spark master: `local[*]`

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

## Results

*To be completed.*

## Discussion

*To be completed.*

## Contributions

| Member | Contributions |
|---|---|
| Eugene Kim | *To be completed* |
| Jack Keeton | *To be completed* |
| Aidan Sanchez | *To be completed* |

## References

- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit Dataset. *Proceedings of the International AAAI Conference on Web and Social Media*, 14, 830–839.
- Apache Spark MLlib Documentation: https://spark.apache.org/docs/latest/ml-guide.html
- HuggingFace Datasets: https://huggingface.co/datasets/fddemarco/pushshift-reddit
