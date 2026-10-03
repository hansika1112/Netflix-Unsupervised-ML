# Netflix Movies & TV Shows Clustering Using Unsupervised Machine Learning

**Subtitle:** Exploratory Data Analysis, NLP-Based Feature Engineering and Unsupervised Content Clustering

## Project overview

This project explores the supplied Netflix Movies and TV Shows catalog and groups titles by text and content metadata. It follows the provided project brief and presentation deck, including EDA, conservative data cleaning, TF-IDF/Bag-of-Words preprocessing, K-Means, Agglomerative Clustering, internal model comparison, cluster interpretation, stakeholder insights, and a video presentation plan.

The primary question is: **Can Netflix titles be grouped into meaningful clusters based on textual and content-related characteristics?** This is an unsupervised clustering project. It does not predict ratings, classify genres, or implement a recommendation engine.

## Dataset and scope

The supplied file is stored at `data/raw/netflix_titles.csv`. The inspected schema contains 7,787 rows and 12 columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.

The brief and presentation describe a dataset of content available as of 2019. No genuine 2019 snapshot was supplied or independently verified. The supplied CSV has `date_added` values from 2008-01-01 through 2021-01-16, including 2,125 rows added after 2019. It is not filtered and is not presented as an official 2019 catalog: results describe the supplied CSV as provided. This is a material source-scope limitation for historical claims.

## Business problem and objectives

Netflix's catalog spans formats, genres, countries, ratings, and release periods. Descriptive segments may help stakeholders explore catalog organization, content mix, and metadata patterns. Objectives include:

- Explore Movies vs TV Shows and how additions vary over time.
- Understand content represented across countries and categories.
- Create interpretable text-based content groupings.
- Compare at least two clustering methods and describe uncertainty.
- Translate technical findings into possible catalog, regional, acquisition, and discovery use cases.
- Demonstrate how clustering could be preprocessing for future recommendation research; this project itself is not a complete recommendation system.

## Technologies and libraries

- Python, pandas, NumPy
- Matplotlib and Seaborn
- scikit-learn: TfidfVectorizer, CountVectorizer, TruncatedSVD, KMeans, AgglomerativeClustering, and cluster metrics
- SciPy: chi-square/Mann–Whitney tests and hierarchical linkage
- Joblib for saving fitted artifacts
- Jupyter Notebook / Google Colab

Install the project dependencies with `pip install -r requirements.txt`.

## Workflow

The notebook makes all seven mandatory stages explicit:

1. **Data Cleaning & Preparation** — inspect the actual schema, missingness, duplicates, whitespace, and data quality; preserve incomplete metadata and legitimate unusual titles.
2. **Exploratory Data Analysis** — examine content type, addition/release years, ratings, countries, genres, durations, directors, cast, and relationships.
3. **Problem Statement & Business Objective** — define clustering and stakeholder objectives.
4. **Model Building** — document model design and train K-Means and Agglomerative clustering after feature construction.
5. **Hypothesis Testing** — test catalog type by date-added era and primary country group using chi-square tests, and compare release-year distributions using Mann–Whitney U.
6. **Feature Engineering & Pre-processing** — create metadata counts and combined text, normalize conservatively, vectorize with TF-IDF and Bag of Words, and reduce TF-IDF with TruncatedSVD.
7. **Storytelling & Video Presentation** — translate measured results into a stakeholder narrative and a timed 10½-minute presentation plan.

### Data cleaning choices

Exact duplicate rows are removed if present; the supplied file contains none. Text columns are stripped and repeated whitespace is collapsed. Missing director/cast/country/date/rating values remain missing in the cleaned data; they are represented as empty text only in combined NLP features and as `Unknown` only where a chart needs a visible category. No rows are dropped solely for missing metadata. `duration` is parsed separately by type because Movie values are minutes and TV Show values are seasons.

### EDA outputs

Figures are saved to `outputs/figures/` and source summaries to `outputs/tables/`. Visuals cover content type, date-added and release-year patterns, ratings by type, top countries/categories/directors/cast, country/type mix, and type-aware duration.

## Hypotheses and statistical results

Significance level: α = 0.05. These tests describe associations within the supplied file; they do not show causation or business impact.

| Hypothesis | Test | Observed result | Interpretation |
|---|---|---|---|
| Content type is associated with date-added era | Pearson chi-square; Cramér's V=0.0285 | χ²=6.311, df=2, raw p=0.0426, Holm p=0.0426 | Reject independence at α=0.05, but association is very small by Cramér's V. TV Show shares are 21.1%, 32.8%, and 30.4% across the three eras. This test does not test a monotonic trend, and these shares do not show a steady increase. |
| Type mix differs by first-listed country group | Pearson chi-square; Cramér's V=0.2947 | χ²=676.376, df=10, raw p≈7.39×10⁻¹³⁹, Holm p≈2.22×10⁻¹³⁸ | Reject independence. Association is larger than the date-era association, but is descriptive; each title contributes once and first-listed country is a limited proxy for co-productions. |
| Release-year distributions differ by type | Two-sided Mann–Whitney U; rank-biserial r=−0.2797 (Movie vs TV) | U=4,666,841.5, raw p≈7.51×10⁻⁸⁸, Holm p≈1.50×10⁻⁸⁷ | Reject equal distributions. Median release years are 2016 for Movies and 2018 for TV Shows. The effect size describes rank separation; neither it nor statistical significance implies causation or business impact. |

All chi-square tests had minimum expected cell counts above five. Holm family-wise adjusted p-values across the three planned tests are reported alongside the original raw p-values. Statistical significance and practical magnitude are treated separately; these tests describe association and do not establish causation. The notebook prints observed tables and detailed values.

## Feature engineering and text representation

`combined_text` uses the available `title`, `director`, `cast`, `country`, `listed_in`, and `description` fields. Text is lowercased in an NLP-specific copy, punctuation is replaced with spaces while Unicode letters/digits are preserved, and whitespace is normalized. TF-IDF uses English stop words and unigrams/bigrams; a unigram CountVectorizer is retained as a Bag-of-Words comparison. The TF-IDF matrix is sparse (30,000 features, about 0.142% density in this run) and is not densified. TruncatedSVD produces 100 components for modeling/visualization; StandardScaler is not applied because the primary vectors are TF-IDF weighted and row-normalized.

## Algorithms, selection, and evaluation

- **K-Means:** candidate K=2–10; inertia and sampled cosine silhouette are compared. The selection rule picks the smallest K within 1% of the best silhouette. The selected K was **9**; the best sampled silhouette was 0.1783 at K=9 (K=10 was within the 1% band). Inertia decreased smoothly, so the silhouette/tie rule is reported explicitly rather than claiming a sharp elbow.
- **Agglomerative Clustering:** Ward/Euclidean and average/cosine linkages were compared over nearby cluster counts on a reproducible 1,500-title sample. Ward with 10 clusters had the highest tested sampled cosine silhouette (0.1359). The dendrogram uses a further 300-title sample for readability and to avoid quadratic full-data cost.
- **Original selected configurations:** K-Means K=9 scored cosine silhouette 0.1769, Davies–Bouldin 2.9533, and Calinski–Harabasz 52.697 on the 1,500-title evaluation sample. Agglomerative Ward K=10 scored 0.1359, 3.1164, and 42.087 on that same sample. This keeps the independent selection outcomes visible and is not a same-K algorithm comparison.
- **Matched-K comparison:** at K=9 on every row of the same 1,500-title sample, K-Means scored cosine silhouette 0.1769, Davies–Bouldin 2.9533, and Calinski–Harabasz 52.697; Agglomerative Ward scored 0.1256, 3.0380, and 42.971. Silhouette favors K-Means; DB and CH are mixed in this matched-K comparison. These internal metrics do not establish semantic truth or business superiority.

Cluster assignments, sizes, approximate centroid terms, cluster-mean TF-IDF terms, representative titles, and metadata profiles are saved in `outputs/tables/`. Each cluster description is assigned after inspecting its actual terms, metadata, and representative examples. Overlapping clusters and broad mixed groups are described conservatively.

## Current business takeaways

- The supplied file contains 5,377 Movies and 2,410 TV Shows; this is a composition count, not evidence of historical strategic focus.
- Content-type mix differs across date-added eras statistically, but the TV Show share peaks in 2015–2017 and is lower in 2018–2021. Avoid claiming a steady rise from these data.
- Country/type association is strong in the supplied file, but country fields are incomplete and multi-valued.
- Some clusters are category-led (including documentary, family, kids-TV, stand-up, and independent-film groupings); broader clusters remain mixed. Use these as candidate navigation/segmentation labels for review.
- Similarity describes metadata and synopsis language, not audience preference or content quality.

## Limitations and future scope

Catalog snapshot provenance differs from the brief's 2019 framing. Date-added data do not fully represent commissioning, acquisition, or removals. Metadata are incomplete, there are no audience labels or engagement data, internal clustering metrics are modest, and the hierarchy is sampled. Future work should confirm the intended snapshot, test alternative field weighting and dimensions, and ask domain reviewers to assess cluster coherence. Any future recommender evaluation would require separately scoped user interaction data and evaluation design.

## Runtime reproducibility

The executed notebook records the actual runtime versions for Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and joblib in `outputs/tables/runtime_versions.json`; the same metadata are embedded in saved model artifacts. `requirements.txt` lists practical minimum dependency versions rather than claiming these are the versions used for this run.

## Project structure

```text
netflix-unsupervised-ml/
├── data/
│   ├── raw/netflix_titles.csv
│   └── processed/netflix_titles_cleaned.csv
├── notebooks/
│   ├── build_notebook.py
│   └── netflix_clustering.ipynb
├── src/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── hypothesis_testing.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── model_runner.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/netflix_clustering.ipynb
```

Run notebook cells from top to bottom. The notebook loads the local file from `data/raw/`, cleans and writes `data/processed/netflix_titles_cleaned.csv`, regenerates figures/tables, fits the models, and saves model artifacts. To rebuild the notebook structure from the maintained cell builder, run `python notebooks/build_notebook.py`.

## Presentation and submission note

The supplied slide deck requests an `.ipynb` and a recorded video in a Drive folder shared with everyone, with a video longer than 7 minutes and an approximate 20–30 minute maximum. It lists 12 Oct 2026 for the two deliverables and 23 Oct 2026 for the Google Form; confirm course logistics. A separate ML submission template is mentioned but was not included among supplied files. The notebook and timed presentation plan are aligned to the visible rubric and can be transferred if the template is provided.
