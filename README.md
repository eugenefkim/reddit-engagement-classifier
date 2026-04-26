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

> **Note:** The module versions below (`cpu/0.15.4`, `gcc/10.2.0`) are specific to the SDSC Expanse cluster Python 3.6 settings. Data loading instructions must be configured based on working environment.

The dataset is hosted on HuggingFace and will be downloaded directly to the cluster for this repository structure. Raw data files are gitignored.

To download the data, open and run the [`notebooks/data_download.ipynb`](notebooks/data_download.ipynb) notebook sequentally from your JupyterLab session on Expanse, following the notebook instructions. This must be run on a compute node (not the login node) to avoid memory limitations, and should be run using Jupyter Notebook settings. 

The dataset is downloaded into `data/raw/`, approximately 89GB of 218 Parquet files. The notebook loads data directly from `data/raw/` using a relative path.

### SDSC Expanse Setup and SparkSession Configuration

> [!WARNING]
> Currently SparkSession is defaulting to Local and the below configuration is set to account for this. Edit this section with the proper SparkSession configuration once distributed computing is configured.

For our exploratory analysis, we use a 16-core, 128GB memory node. Driver is allocated 8GB to account for Local default Spark Master configuration. Executor instances are set to 6 to leave headroom for system overhead, rather than the strict `Total Cores - 1 = 15` formula. Executor memory would be calculated as `(Total Memory - Driver Memory) / Executor Instances = (128 - 8) / 6 ≈ 20GB`, conservatively set to 16GB to account for Spark memory overhead and off-heap usage. However, executor configurations are not being set due to the Local SparkSession defaulting that is currently occurring. 

```python
from pyspark.sql import SparkSession

# 16 cores, 128GB total memory
# Executor instances = Total Cores - 1 = 15
# Executor memory = (Total Memory - Driver Memory) / Executor Instances
#                 = (128 - 8) / 6 ≈ 16GB (6 executors used to leave headroom)
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

This configuration is acceptable for EDA but will be modified once distributed computation is configured. 

### SparkUI Screenshot
<img width="853" height="423" alt="SparkUI_Screenshot_Local" src="https://github.com/user-attachments/assets/42f747a7-9c31-4201-8586-89919649a849" />

## Notebook

The Milestone 2 EDA and data exploration notebook is located at: [`notebooks/Milestone2_Pushshift.ipynb`](notebooks/Milestone2_Pushshift.ipynb)

## Preprocessing Plan

Preprocessing will be implemented in Milestone 3 using PySpark DataFrame operations. A key consideration in preprocessing is Pushshift's frequent usage of sentinel strings (e.g., `[deleted]` or `[removed]`) and empty strings instead of SQL NULLs, a naive null check will completely miss many true missing values. Also worth noting is the prevalence of suspected bot authors. We consider all these factors in our preprocessing plan, which is as follows:

### Missing Values
- `title`: No missing values will be included in model DF. Rows with missing titles (`title_len` = 0) will be dropped as they represent records with no usable signal.
- `selftext`: Missing values are valid and indicate link posts; missing values will be flagged via a binary `has_text` indicator column. These include `[removed]` and `[deleted]` sentinel string values.
- `author`: Rows with deleted (`[deleted]`) or anonymous authors will be filtered out as they cannot contribute to author history features. We also potentially plan to filter out suspected bot authors ( `AutoModerator`, `AutoNewsAdmin`, etc.). We will flag any author that posts at an inhuman frequency by defining a threshold of reasonable number posts per day and consider filtering them out in Milestone 3. 
- `subreddit` and `subreddit_id`: All null values with be dropped (EDA discovered ~306K null values) as these cannot contribute to per-subreddit features. 
- `score` / `num_comments`: Rows with null values in either column will be dropped as these are required for label generation. Additionally, negative values for `num_comments` will be filtered out as these represent invalid records.

### Duplicate Posts
All posts with duplicate IDs will be deduplicated. This number is small but must be handled.

### Class Imbalance
We expect the majority of posts to fall into the low-engagement class. We will address this using PySpark MLlib's `weightCol` parameter to assign inverse-frequency class weights during model training, giving the minority classes (viral, crowd-pleaser, debate-starter) proportionally higher influence on the loss function.

### Label Generation
Labels will be assigned using `pyspark.sql.Window` functions to compute per-subreddit 50th percentile thresholds for `score` and `num_comments`, then applying a four-way conditional to assign each post its engagement archetype. We will also consider implementing a more straight-forward two-tier label of high vs. low engagement for a cleaner model and compare to our four-tier label model. 

### Feature Engineering
All features will be derived from pre-publication information only to prevent data leakage. Transformations include:

- **Title features**: character count, word count, question/exclamation presence, sentiment polarity, readability — extracted using PySpark UDFs wrapping NLTK/TextBlob
- **Post type**: binary `is_text_post` flag derived from `selftext`; selftext length for text posts
- **Author history**: mean score, posting frequency, subreddit diversity — computed via `groupBy().agg()` window aggregations across the full dataset
- **Subreddit context**: post volume, median score, median comment count — computed via `groupBy().agg()` per subreddit
- **Temporal**: hour of day and day of week extracted from `created_utc` via `pyspark.sql.functions.from_unixtime`

### Scaling and Encoding
- All continuous features will be standardized using MLlib's `StandardScaler`
- Categorical features (`subreddit`, `author`) will not be directly one-hot encoded due to high cardinality (millions of unique authors, hundreds of thousands of subreddits). Instead, each is represented through aggregated numerical features that capture their unique behavioral signals, e.g., author history features for `author` and community context features for `subreddit`
- The final feature vector will be assembled using MLlib's `VectorAssembler`

### Spark Operations for Preprocessing
Key PySpark operations used:
- `df.cache()` - prevent repeated and expensive full df scans from repeated `df.count()` calls
- `df.filter()` — remove null/invalid rows
- `df.fillna()` — impute nulls in selftext
- `Window.partitionBy()` + `percentile_approx()` — per-subreddit quantile thresholds
- `df.groupBy().agg()` — author and subreddit feature aggregations
- `df.withColumn()` + UDFs — title linguistic feature extraction
- `VectorAssembler` — final feature vector construction
- `StandardScaler` — feature normalization

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
