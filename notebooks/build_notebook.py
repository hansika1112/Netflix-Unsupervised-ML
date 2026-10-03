"""Build the documented, executable project notebook from reusable src modules."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("netflix_clustering.ipynb")

cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def py(source):
    cells.append(nbf.v4.new_code_cell(source))


md(r"""# Netflix Movies & TV Shows Clustering Using Unsupervised Machine Learning
### Exploratory Data Analysis, NLP-Based Feature Engineering and Unsupervised Content Clustering

**Project type:** Unsupervised Machine Learning · EDA · NLP/Text Mining · Clustering · Business Analytics  
**Primary question:** Can Netflix titles be grouped into useful content segments using text and content metadata?

This notebook is organized into the seven required project stages. It uses the supplied CSV and the project brief/deck. The brief describes a 2019 catalog snapshot, while the supplied CSV contains `date_added` values later than 2019; analyses therefore describe the supplied file as provided and flag that scope mismatch. No source rows are silently removed for having incomplete metadata.
""")

md("## Stage 1 — Data Cleaning & Preparation\n\nWe first inspect the real file and schema. Cleaning decisions are evidence-based and preserve incomplete but potentially informative titles.")
md("### 1. Project setup and reproducibility")
py(r'''from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Saved project figures render identically in Colab and local Jupyter.
from IPython.display import display, Markdown, Image

_cwd = Path.cwd()
_candidates = [_cwd, _cwd.parent, _cwd / "netflix-unsupervised-ml", _cwd.parent / "netflix-unsupervised-ml"]
PROJECT_ROOT = next((p.resolve() for p in _candidates if (p / "src").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Open this notebook from the project root or notebooks folder.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import load_titles
from data_cleaning import inspect_schema, clean_titles
from eda import run_eda
from hypothesis_testing import run_hypothesis_tests
from feature_engineering import make_text_features, add_metadata_features
from preprocessing import vectorize_text
from model_runner import run_models

RAW_PATH = PROJECT_ROOT / "data/raw/netflix_titles.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/netflix_titles_cleaned.csv"
FIGURES = PROJECT_ROOT / "outputs/figures"
TABLES = PROJECT_ROOT / "outputs/tables"
MODELS = PROJECT_ROOT / "outputs/models"
for folder in [FIGURES, TABLES, MODELS, PROCESSED_PATH.parent]:
    folder.mkdir(parents=True, exist_ok=True)

raw_df = load_titles(RAW_PATH)
print(f"Loaded {RAW_PATH.name} | shape={raw_df.shape}")''')

md("### 2. Required schema inspection")
py(r'''schema = inspect_schema(raw_df)
print("Shape:", schema["shape"])
print("Columns:", schema["columns"])
display(raw_df.head())
display(raw_df.tail())
inspection = pd.DataFrame({
    "dtype": raw_df.dtypes.astype(str),
    "missing_count": raw_df.isna().sum(),
    "missing_pct": (raw_df.isna().mean() * 100).round(2),
    "unique_values": raw_df.nunique(dropna=False),
}).sort_index()
display(inspection)
print("Exact duplicate rows:", int(raw_df.duplicated().sum()))
print("Numeric columns:", raw_df.select_dtypes(include="number").columns.tolist())
print("String columns:", raw_df.select_dtypes(include=["str", "object"]).columns.tolist())
print("Low-cardinality unique values:")
for col in raw_df.columns:
    if raw_df[col].nunique(dropna=False) <= 20:
        print(f"  {col}: {raw_df[col].dropna().unique().tolist()}")
    else:
        print(f"  {col}: {raw_df[col].nunique(dropna=False):,} unique; sample={raw_df[col].dropna().head(3).tolist()}")''')

md("### 3. Data dictionary from the observed schema")
py(r'''data_dictionary = pd.read_csv(TABLES / "data_dictionary.csv")
display(data_dictionary)
print("All values are stored as strings except release_year in this CSV. Duration is a mixed-unit string (minutes for Movies, seasons for TV Shows).")''')

md(r"""### 4. Missing values, duplicates, and conservative cleaning decisions

| Field(s) | Observed issue | Decision | Reason |
|---|---:|---|---|
| `director` | substantial missingness | Preserve missing | Director may not be supplied; filling a fabricated name adds false evidence. Empty text is used only in vector construction. |
| `cast`, `country` | missing metadata | Preserve; display as `Unknown` only in summaries where useful | Missing credits/country attribution do not make the title unusable. |
| `date_added`, `rating` | a small number missing | Preserve in source; date analyses omit only rows without a usable date, ratings display an Unknown bucket | Avoid inventing dates or rating categories. |
| `description`, `duration`, `title`, `listed_in` | complete in supplied file | Retain | These are valuable content fields. |
| exact duplicate rows | checked below | Remove exact duplicates only, if any | Near-duplicate or similar titles can be legitimate distinct records. |
| string whitespace | inconsistent edge/repeated whitespace | Strip ends and normalize repeated whitespace; preserve capitalization and meaningful punctuation in source columns | Conservative text normalization; lowercase is applied only to the NLP copy. |

The `show_id` field is an identifier, not a clustering feature. Release-year extremes and short/long durations are reviewed as possible legitimate cases, not dropped by a generic outlier rule.
""")
py(r'''clean_df, cleaning_impact = clean_titles(raw_df)
clean_df.to_csv(PROCESSED_PATH, index=False)
cleaning_impact.to_csv(TABLES / "cleaning_impact.csv")
display(cleaning_impact)
print(f"Before: {raw_df.shape}; after: {clean_df.shape}; exact duplicates after cleaning: {clean_df.duplicated().sum()}")
print("Missing counts retained:")
display(clean_df.isna().sum().to_frame("missing_after_cleaning"))
print("Invalid content types:", sorted(set(clean_df["type"].dropna()) - {"Movie", "TV Show"}))
movie_unit_errors = int((clean_df["type"].eq("Movie") & ~clean_df["duration"].str.endswith("min")).sum())
tv_unit_errors = int((clean_df["type"].eq("TV Show") & ~clean_df["duration"].str.contains("Season", case=False)).sum())
print("Duration-unit mismatches — Movie:", movie_unit_errors, "TV Show:", tv_unit_errors)''')

md("## Stage 2 — Exploratory Data Analysis (EDA)\n\nAll charts are generated from the cleaned supplied dataset and saved under `outputs/figures/`. Each interpretation is printed with the chart.")
py(r'''eda_results = run_eda(clean_df, FIGURES, TABLES)
type_counts = clean_df["type"].value_counts()
added = pd.to_datetime(clean_df["date_added"], errors="coerce")
added_type = pd.read_csv(TABLES / "titles_added_by_year_and_type.csv", index_col=0)
display(type_counts.rename("titles").to_frame())
print("Added-date range:", added.min(), "to", added.max())''')

md("### 5. Movies vs TV Shows")
py(r'''display(Image(filename=str(FIGURES / "content_type_distribution.png")))
movie_n, tv_n = int(type_counts.get("Movie", 0)), int(type_counts.get("TV Show", 0))
display(Markdown(f"**INSIGHT:** The supplied file contains {movie_n:,} Movies and {tv_n:,} TV Shows. Movies are {movie_n/(movie_n+tv_n):.1%} of these rows. This is a catalog composition snapshot, not by itself evidence of a change over time.\n\n**BUSINESS RELEVANCE:** Catalog mix can inform how stakeholders organize title discovery and compare formats, while trend claims require date-aware analysis."))''')

md("### 6. Content added over time")
py(r'''display(Image(filename=str(FIGURES / "titles_added_by_year.png")))
display(Image(filename=str(FIGURES / "titles_added_by_year_and_type.png")))
year_totals = added.dt.year.value_counts().sort_index()
peak_year = int(year_totals.idxmax())
display(Markdown(f"**INSIGHT:** The supplied file records additions from {added.min():%Y-%m-%d} through {added.max():%Y-%m-%d}; the largest recorded year is {peak_year} ({int(year_totals.max()):,} rows). These counts are based on this file's `date_added` values.\n\n**BUSINESS RELEVANCE:** Addition patterns can help frame catalog-planning questions. This retrospective file cannot establish commissioning intent or explain removals."))''')

md("### 7. Release year and content type")
py(r'''display(Image(filename=str(FIGURES / "release_year_distribution.png")))
display(Image(filename=str(FIGURES / "release_year_by_content_type.png")))
display(Markdown(f"**INSIGHT:** Release years in the file range from {clean_df.release_year.min()} to {clean_df.release_year.max()}; {int((clean_df.release_year < 1970).sum())} titles predate 1970. Older entries are retained for review.\n\n**BUSINESS RELEVANCE:** A broad release-year mix can matter when stakeholders plan catalog organization across newer releases and library titles."))''')

md("### 8. Ratings and content type")
py(r'''display(Image(filename=str(FIGURES / "rating_distribution.png")))
rating_counts = clean_df["rating"].fillna("Unknown").value_counts()
display(rating_counts.head(10).to_frame("titles"))
display(Markdown(f"**INSIGHT:** The most frequent rating label in the supplied catalog is **{rating_counts.index[0]}** ({int(rating_counts.iloc[0]):,} rows, including any Unknown bucket if applicable).\n\n**BUSINESS RELEVANCE:** Rating mix can support age-appropriate catalog filters and content portfolio summaries. It is not a measure of audience response."))''')

md("### 9. Country distribution and country vs content type")
py(r'''display(Image(filename=str(FIGURES / "country_distribution.png")))
display(Image(filename=str(FIGURES / "country_by_content_type.png")))
display(Image(filename=str(FIGURES / "country_type_heatmap.png")))
country_top = pd.read_csv(TABLES / "top_country.csv").iloc[0]
display(Markdown(f"**INSIGHT:** The most frequently listed country mention is **{country_top['country']}** ({int(country_top['title_mentions']):,} title mentions). Titles can list multiple countries, so these are mentions and need not sum to the number of rows. The heatmap compares observed type mix across the most represented countries.\n\n**BUSINESS RELEVANCE:** Regional catalog views can guide further market analysis. Multi-country credits and missing country metadata limit country-level comparisons."))''')

md("### 10. Genre/category distribution")
py(r'''display(Image(filename=str(FIGURES / "genre_distribution.png")))
genre_top = pd.read_csv(TABLES / "top_listed_in.csv").iloc[0]
display(Markdown(f"**INSIGHT:** The most common listed category is **{genre_top['listed_in']}** ({int(genre_top['title_mentions']):,} title mentions). `listed_in` is multi-label, so one title may contribute to several categories.\n\n**BUSINESS RELEVANCE:** Multi-label categories can make catalog navigation and segment-level content analysis more useful."))''')

md("### 11. Duration, directors, and cast")
py(r'''display(Image(filename=str(FIGURES / "movie_duration_minutes.png")))
display(Image(filename=str(FIGURES / "tv_show_duration_seasons.png")))
display(Image(filename=str(FIGURES / "director_distribution.png")))
display(Image(filename=str(FIGURES / "cast_distribution.png")))
duration_summary = eda_results["duration_medians"]
display(Markdown(f"**INSIGHT:** Median Movie duration is {duration_summary['Movie_minutes']:.0f} minutes; median TV Show duration is {duration_summary['TV_Show_seasons']:.0f} seasons. These are separate measurements on separate plots and are not numerically comparable. Director/cast metadata are long-tailed and incomplete.\n\n**BUSINESS RELEVANCE:** Format-aware duration summaries support catalog filters, while director and cast credits can enrich discovery when supplied."))''')

md("## Stage 3 — Problem Statement & Business Objective")
md(r"""### 12. Problem statement

Netflix has a large catalog of Movies and TV Shows. **Can titles be grouped into meaningful clusters based on textual and content-related characteristics?**

This project selects **unsupervised clustering**. No ground-truth segment labels are available, and the objective is to explore similarities rather than predict ratings, classify genres, or build a recommendation engine.

### 13. Business objectives

1. Explore catalog segments and content discovery patterns.
2. Support catalog organization and genre/content segmentation.
3. Compare country and format patterns for regional analysis.
4. Provide descriptive evidence for content-strategy and acquisition discussions.
5. Produce possible audience-oriented groupings for later investigation.
6. Offer a clustering/segmentation preprocessing concept for future recommendation research; this project itself is **not** a recommendation system.

### 14. Business questions

- What content types, release periods, ratings, countries, and categories appear in this supplied catalog?
- Do observed content-type shares vary across date-added periods and listed countries?
- Can text similarities produce interpretable title groups?
- How stable and useful are K-Means and agglomerative groupings under internal metrics?
""")

md("## Stage 4 — Model Building (Design)\n\nThe model implementation follows feature construction in Stage 6 so the notebook remains executable in the requested analytical order. We compare K-Means with Agglomerative Clustering, select K from measured evidence, and report internal metrics without claiming a ground-truth accuracy score.")

md("## Stage 5 — Hypothesis Testing")
md(r"""The significance threshold is **α = 0.05**. Chi-square tests require independent observational units and expected cell counts large enough for the asymptotic approximation. The country test uses one first-listed country per title, top 10 countries plus `Other`, so each title appears once. Non-significant/significant results do not establish causation or business value by themselves.

### H1 — Type mix by date-added era

- **H₀:** Content type and broad date-added era are independent.
- **H₁:** Content-type distribution differs by era.
- A chi-square association test is paired with observed TV Show shares to examine direction. A significant association is not equivalent to a monotonic increase.
- The date-era chi-square test treats era as categorical; it tests association between content type and era, not an ordered/monotonic increase in TV Shows. The observed TV shares must be read directly and are not steadily increasing here.

### H2 — Type mix by primary listed country

- **H₀:** Content type is independent of primary listed country group.
- **H₁:** Content-type distribution differs across these country groups.
- This first-listed-country operationalization is incomplete for co-productions and is descriptive.

### H3 — Release-year distributions by type

- **H₀:** Movie and TV Show release-year distributions are equal.
- **H₁:** The distributions differ.
- The two-sided Mann–Whitney U test avoids a normality assumption for the skewed, discrete year values; titles are treated as independent rows.

For each association, the notebook reports effect size as well as p-value: Cramér's V for categorical chi-square tests, and rank-biserial correlation for the Mann–Whitney comparison. The three planned p-values are also shown with Holm family-wise adjustment; raw values remain visible. Statistical significance is distinct from practical importance, and no test establishes causation.
""")
py(r'''hypotheses = run_hypothesis_tests(clean_df, alpha=0.05)
for result in hypotheses:
    effect = (f"Cramer's V={result['cramers_v']:.4f}" if "cramers_v" in result else
              f"rank-biserial (Movie vs TV)={result['rank_biserial_movie_vs_tv']:.4f}")
    print(f"{result['hypothesis']}\nTest: {result['test']} | statistic={result['statistic']:.4g} | raw p={result['p_value']:.4g} | Holm p={result['p_value_holm']:.4g} | alpha={result['alpha']:.2f} | raw reject={result['reject_null']} | Holm reject={result['reject_null_holm']} | effect size={effect}")
    if result.get("assumption_min_expected") is not None:
        print(f"Minimum expected cell count: {result['assumption_min_expected']:.2f}")
    if result.get("direction_values"):
        print("Observed TV Show share by era:", {k: f"{v:.1%}" for k,v in result["direction_values"].items()})
    if result.get("median_movie_year") is not None:
        print("Release-year medians: Movie=", result["median_movie_year"], "TV Show=", result["median_tv_year"])
    print("Interpretation:", result["interpretation"], "\n")
with open(TABLES / "hypothesis_test_results.json", "w", encoding="utf-8") as f:
    json.dump(hypotheses, f, indent=2, ensure_ascii=False)
display(pd.DataFrame([{k: r.get(k) for k in ["hypothesis","test","statistic","degrees_of_freedom","p_value","p_value_holm","alpha","reject_null","reject_null_holm","cramers_v","rank_biserial_movie_vs_tv","assumption_min_expected"]} for r in hypotheses]))''')
py(r'''display(Markdown(
    f"**Practical magnitude:** date-era Cramér's V={hypotheses[0]['cramers_v']:.4f}; primary-country Cramér's V={hypotheses[1]['cramers_v']:.4f}; release-year rank-biserial effect={hypotheses[2]['rank_biserial_movie_vs_tv']:.4f} (negative means Movie years tend lower than TV Show years under this ordering). The date-era result is an association test only, and its observed TV shares are not monotonic. P-values address compatibility with the null under test assumptions; effect sizes describe magnitude, and neither establishes business impact."))''')

md("## Stage 6 — Feature Engineering & Pre-processing")
md("### 15. Metadata feature engineering")
py(r'''featured_df = add_metadata_features(make_text_features(clean_df))
metadata_features = [c for c in ["year_added","country_count","category_count","cast_member_count","director_count","duration_value","duration_unit"] if c in featured_df]
display(featured_df[["show_id","title","type","release_year"] + metadata_features].head())
print("Text fields used when present:", [c for c in ["title","director","cast","country","listed_in","description"] if c in featured_df.columns])''')

md("### 16. Combined text and preprocessing")
md(r"""`combined_text` joins available `title`, `director`, `cast`, `country`, `listed_in`, and `description`. The fields provide names, credits, geography, categories, and synopsis language that can express content similarity. Missing values become empty only in this derived NLP representation. The normalized copy is lowercased, punctuation is replaced with spaces while Unicode letters/digits are preserved, and whitespace is collapsed. English stop words are handled by the vectorizers. Source metadata remains unchanged.

Numeric metadata (release year, counts, duration) is retained for interpretation/EDA and not appended to the text distance space. This avoids mixing arbitrary numeric scale with sparse text similarity. `StandardScaler` is therefore not applied to the primary representation.
""")
py(r'''display(featured_df[["title","combined_text","normalized_text"]].head(3))
print("Rows with empty normalized text:", int(featured_df.normalized_text.eq("").sum()))''')

md("### 17. TF-IDF and Bag of Words")
md(r"""**Term Frequency (TF)** counts a term within a title document. **Inverse Document Frequency (IDF)** downweights terms common across many documents. **TF-IDF** combines both so a term receives higher weight when it is informative for a title and less weight when ubiquitous. A sparse matrix makes cosine similarity practical without densifying the vocabulary matrix.

CountVectorizer is also run as a Bag-of-Words baseline: it records token counts, without the same common-term downweighting. TF-IDF is the primary representation.
""")
py(r'''tfidf, tfidf_vectorizer, bow, bow_vectorizer = vectorize_text(featured_df["normalized_text"])
representation_summary = pd.DataFrame([
    {"representation":"TF-IDF (1-2 grams)","rows":tfidf.shape[0],"features":tfidf.shape[1],"nonzeros":tfidf.nnz,"density":tfidf.nnz/(tfidf.shape[0]*tfidf.shape[1])},
    {"representation":"Bag of Words (unigrams)","rows":bow.shape[0],"features":bow.shape[1],"nonzeros":bow.nnz,"density":bow.nnz/(bow.shape[0]*bow.shape[1])},
])
display(representation_summary)
print("Sparse TF-IDF matrix:", tfidf.shape, "| density:", f"{tfidf.nnz/(tfidf.shape[0]*tfidf.shape[1]):.4%}")''')

md("### 18. Numerical and dimensionality preprocessing")
md(r"""TruncatedSVD reduces the sparse TF-IDF matrix to a dense 100-component representation for clustering and visualization. Rows are L2-normalized after projection so Euclidean clustering is not dominated by vector magnitude. The 100-component space captures only part of original TF-IDF variability; the 2D map is a qualitative projection, not proof of clean separation. The original sparse TF-IDF matrix stays sparse.
""")

md("### 19. K-Means: selecting K, elbow, and silhouette")
md("#### Stage 4 continued — Model implementation")
md(r"""Candidate K values are fitted over 2–10. Inertia must decrease as K grows, so it is interpreted with the elbow shape and a sampled cosine silhouette score (higher is better). The selection rule used here chooses the smallest K within 1% of the best sampled silhouette. This is a transparent complexity trade-off; it does not claim one objectively true number of segments.
""")
py(r'''model_results = run_models(clean_df, PROJECT_ROOT, k_values=range(2,11),
                           agglomerative_sample_size=1500, random_state=42)
k_selection = model_results["k_selection"]
selected_k = model_results["selected_k"]
display(k_selection)
display(Image(filename=str(FIGURES / "elbow_and_silhouette.png")))
print("Selected K by stated rule:", selected_k)
display(Markdown(f"**INSIGHT:** The selected value is K={selected_k}, based on the observed inertia curve and candidate sampled silhouette scores.\n\n**BUSINESS RELEVANCE:** A documented selection rule makes segment counts reviewable; stakeholders should treat segments as exploratory organization aids."))''')

md("### 20. Final K-Means model")
py(r'''labels = model_results["kmeans_labels"]
cluster_sizes = pd.Series(labels).value_counts().sort_index().rename("titles").to_frame()
cluster_sizes["percentage"] = (cluster_sizes["titles"] / len(labels) * 100).round(2)
display(cluster_sizes)
display(Image(filename=str(FIGURES / "cluster_sizes.png")))
print("Assignments saved to:", "outputs/tables/kmeans_cluster_assignments.csv")''')

md("### 21. Agglomerative hierarchical clustering and dendrogram")
md(r"""A complete hierarchical fit on all titles requires quadratic pairwise storage/work and is unnecessary for this exploratory comparison. We use a reproducible 1,500-title sample, evaluate Ward/Euclidean and average/cosine linkage over cluster counts near the selected K, then draw a readable truncated dendrogram from a further 300-title representative subset. Hierarchical assignments are therefore sample assignments, not a full-catalog deployment model.
""")
py(r'''display(model_results["agglomerative_selection"])
display(Image(filename=str(FIGURES / "dendrogram.png")))
display(Image(filename=str(FIGURES / "agglomerative_cluster_visualization.png")))
print("Selected linkage and cluster count:", model_results["selected_linkage"], model_results["hierarchy_k"])
print("Agglomerative sample size:", len(model_results["agglomerative_labels"]))''')

md("### 22. Model evaluation and comparison")
md(r"""Metrics are computed for both methods on the same 1,500-title sample in the normalized SVD space:

- **Cosine silhouette**: cohesion/separation from cosine geometry; higher is better, range −1 to 1.
- **Davies–Bouldin**: average similarity to the most similar competing cluster under Euclidean geometry; lower is better.
- **Calinski–Harabasz**: between-cluster dispersion relative to within-cluster dispersion under Euclidean geometry; higher is better.
- **Inertia**: K-Means within-cluster squared Euclidean error over the full fitted catalog; lower as K rises, so compare only alongside K.

Internal metrics do not establish semantic truth. Mixed text fields, high sparsity, and the SVD projection affect these values.
""")
md("#### Original selected-configuration results (model selection)")
md("This table retains each algorithm's selected configuration. K-Means K is selected by its stated 1% silhouette rule; Agglomerative linkage/K is chosen from its own sampled linkage/K search. Since their K values can differ, this table is not a controlled same-K algorithm comparison.")
py(r'''selected_comparison = model_results["metrics"]
display(selected_comparison)''')

md("#### Matched-K algorithm comparison")
md("For a controlled comparison, both methods use K equal to the selected K-Means K, with labels evaluated on every row of the same reproducible 1,500-title sample. This comparison is reported separately from model selection. Silhouette uses cosine geometry; Davies–Bouldin and Calinski–Harabasz use Euclidean geometry in the same normalized SVD space.")
py(r'''comparison = model_results["same_k_metrics"]
display(comparison)
km_row, ag_row = comparison.iloc[0], comparison.iloc[1]
display(Markdown(
    f"**Matched-K interpretation (K={int(km_row.n_clusters_requested)}):** On the same {int(km_row.n_evaluated):,}-title evaluation sample, K-Means has cosine silhouette {km_row.silhouette_cosine:.3f} vs {ag_row.silhouette_cosine:.3f} for Agglomerative Ward, Davies–Bouldin {km_row.davies_bouldin_euclidean:.3f} vs {ag_row.davies_bouldin_euclidean:.3f} (lower is better), and Calinski–Harabasz {km_row.calinski_harabasz_euclidean:.2f} vs {ag_row.calinski_harabasz_euclidean:.2f} (higher is better). These internal metrics describe this representation and sample only; they do not establish semantic or business superiority. K-Means is used for full-catalog interpretation, while hierarchy provides a sample-based second view."
))''')

md("### 23. Cluster visualization")
py(r'''display(Image(filename=str(FIGURES / "cluster_visualization.png")))
display(Markdown("**INSIGHT:** Each point is a title projected to the first two SVD components, colored by K-Means label. Overlap is expected because only two dimensions are shown.\n\n**BUSINESS RELEVANCE:** This is a visual overview for discussion; it should not be used to claim that titles are perfectly separated."))''')

md("### 24. Cluster interpretation and cluster strategy")
py(r'''profiles = model_results["profiles"]
def descriptive_cluster_name(profile):
    shares = profile["content_type_share_pct"]
    dominant_type = max(shares, key=shares.get)
    leading_category = profile["common_genres"][0] if profile["common_genres"] else "mixed categories"
    return f"{dominant_type}-focused · {leading_category}-led"

for profile in profiles:
    profile["descriptive_name_after_review"] = descriptive_cluster_name(profile)
    print(f"\nCluster {profile['cluster']} — {profile['descriptive_name_after_review']}")
    print("Size:", profile["size"], f"({profile['percentage']:.2f}%)")
    print("Content type share (%):", profile["content_type_share_pct"])
    print("Common countries:", profile["common_countries"])
    print("Common ratings:", profile["common_ratings"])
    print("Common genres:", profile["common_genres"])
    print("K-Means centroid terms (approximate TF-IDF projection):", profile["kmeans_centroid_terms_approx"])
    print("Cluster mean TF-IDF terms:", profile["important_tfidf_terms"])
    print("Representative titles:", profile["representative_titles"])
    member_rows = featured_df.iloc[np.flatnonzero(labels == profile["cluster"])]
    representative_rows = member_rows.loc[member_rows["title"].isin(profile["representative_titles"]), ["title","description"]]
    display(representative_rows.head(5))
    dominant_type = max(profile["content_type_share_pct"], key=profile["content_type_share_pct"].get)
    category = profile["common_genres"][0] if profile["common_genres"] else "mixed-category"
    profile["business_interpretation"] = f"Potential use: use this {category.lower()}-led {dominant_type.lower()} segment for navigation and acquisition discussions; validate title fit and audience utility before operational use."
    print("Characteristics:", f"{dominant_type}-focused; strongest listed category is {category}; terms and example synopses above provide the text evidence.")
    print("Business interpretation:", profile["business_interpretation"])
    display(Image(filename=str(FIGURES / f"cluster_{profile['cluster']}_top_terms.png")))
with open(TABLES / "cluster_profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2)
pd.DataFrame([{k:v for k,v in p.items() if k != "description_samples"} for p in profiles]).to_csv(TABLES / "cluster_profiles.csv", index=False)''')

md(r"""### 25. Model explainability

K-Means assigns a title to the nearest centroid in the reduced SVD space. The notebook shows an approximate inverse-SVD projection of each centroid back to TF-IDF terms, plus average TF-IDF terms across actual assigned members and representative titles nearest to each cluster's mean TF-IDF vector. The explanation is understandable as: a title joins a segment when its weighted words/categories and metadata align with that segment's overall textual profile. The inverse projection is approximate because dimensionality reduction discards information.

Agglomerative clustering joins the closest groups step by step; Ward linkage minimizes increases in within-group squared Euclidean dispersion. The dendrogram summarizes merge distances for the 300-title sample. A high merge distance suggests relatively dissimilar groups being joined, but the tree does not supply semantic names.
""")

md("### 26. Exceptional case analysis")
py(r'''duration_values = pd.to_numeric(clean_df["duration"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
movie_dur = duration_values[clean_df["type"].eq("Movie")]
exception_table = pd.DataFrame([
    {"case":"release_year <= 1950", "count":int((clean_df.release_year <= 1950).sum()), "handling":"Retain; likely legitimate catalog/library entries; inspect titles before treating as errors."},
    {"case":"release_year < 1970", "count":int((clean_df.release_year < 1970).sum()), "handling":"Retain; old release year alone is not a data-quality failure."},
    {"case":"Movies under 60 minutes", "count":int((clean_df.type.eq("Movie") & movie_dur.lt(60)).sum()), "handling":"Retain; short films/specials may be valid. Review title-level cases."},
    {"case":"Movies over 180 minutes", "count":int((clean_df.type.eq("Movie") & movie_dur.gt(180)).sum()), "handling":"Retain; long/special-format titles may be legitimate."},
    {"case":"missing director, cast, or country simultaneously", "count":int(clean_df[["director","cast","country"]].isna().all(axis=1).sum()), "handling":"Retain; sparse credits reduce metadata signal but descriptions remain available."},
    {"case":"missing descriptions", "count":int(clean_df.description.isna().sum()), "handling":"None in the supplied CSV; preserve if present."},
    {"case":"rare rating labels (count <= 5)", "count":int(clean_df.rating.value_counts().le(5).sum()), "handling":"Retain rare labels; labels such as NC-17/UR can be uncommon but are not inherently invalid."},
    {"case":"country mentions appearing once", "count":int((clean_df.country.fillna('').str.split(', ').explode().str.strip().value_counts()==1).sum()), "handling":"Retain rare country mentions; uncommon credits may describe legitimate titles."},
])
display(exception_table)
exception_table.to_csv(TABLES / "exceptional_cases.csv", index=False)
print("Movie duration numeric summary (minutes):")
display(movie_dur.describe().to_frame("minutes"))
print("TV Show season counts:")
display(clean_df.loc[clean_df.type.eq("TV Show"), "duration"].value_counts().head(10).to_frame("titles"))
print("Potential edge examples:")
display(clean_df.loc[clean_df.type.eq("Movie") & movie_dur.lt(60), ["title","release_year","duration","description"]].head(10))
print("Oldest listed releases:")
display(clean_df.nsmallest(5, "release_year")[["title","release_year","type","duration"]])
print("Longest Movie durations:")
display(clean_df.loc[clean_df.type.eq("Movie")].assign(duration_minutes=movie_dur).nlargest(5,"duration_minutes")[["title","release_year","duration","description"]])
print("Rare rating categories:")
rare = clean_df["rating"].value_counts()
display(rare[rare.le(10)].to_frame("titles"))''')

md("## 27. Business insights and stakeholder use cases")
py(r'''business_rows = [
    {"Technical Finding":f"{movie_n:,} Movies vs {tv_n:,} TV Shows in this supplied file", "Business Meaning":"Catalog composition is movie-heavy in row count", "Possible Netflix Use":"Format-aware catalog summaries and discovery surfaces"},
    {"Technical Finding":f"H1 date-era chi-square p={hypotheses[0]['p_value']:.4g}; TV shares={hypotheses[0]['direction_values']}", "Business Meaning":"Type mix differs statistically across broad addition eras, but the shares are not monotonic", "Possible Netflix Use":"Use as a prompt for deeper time-series/catalog-history analysis; avoid a simple 'TV increasingly replaced Movies' claim"},
    {"Technical Finding":f"H2 primary-country association p={hypotheses[1]['p_value']:.4g}", "Business Meaning":"Observed type mix varies across first-listed country groups", "Possible Netflix Use":"Explore regional catalog organization, with co-production/missing-data caveats"},
    {"Technical Finding":f"K-Means selected K={selected_k}; sampled cosine silhouette={comparison.iloc[0]['silhouette_cosine']:.3f}", "Business Meaning":"Text segments show some structure but remain overlapping", "Possible Netflix Use":"Use clusters as candidate catalog tags to review, not final personalization decisions"},
    {"Technical Finding":"Documentary, family, kids-TV, stand-up, international and broad drama/TV groups appear in profiles", "Business Meaning":"Some clusters are category/format-led and some are broad", "Possible Netflix Use":"Improve exploratory catalog browsing and inform content-segment discussions"},
    {"Technical Finding":"Text clusters are built from title, credits, country, categories, and synopsis", "Business Meaning":"Similarity reflects available metadata wording, not viewer behavior", "Possible Netflix Use":"Potential preprocessing for future recommender research; not a recommender itself"},
]
business_table = pd.DataFrame(business_rows)
display(business_table)
business_table.to_csv(TABLES / "business_insights.csv", index=False)''')

md("### 28. Limitations and future scope")
md(r"""**Limitations**

- The supplied file extends to `date_added` in 2021, while the brief frames the data as available by 2019; source-snapshot provenance should be confirmed before making 2019-specific claims.
- `date_added` in this catalog is not a complete record of content acquisition, removals, or commissioning; associations do not establish strategic intent.
- Missing director (30.68%), cast (9.22%), country (6.51%), rating (0.09%), and date-added (0.13%) metadata remain visible. Country/category fields are multi-valued and some analyses use first-listed or exploded counts.
- Clustering is unsupervised; no ground-truth labels or user engagement data are available. Internal scores are modest/overlapping and depend on TF-IDF/SVD/model choices.
- Hierarchical fitting is sampled (1,500 titles; dendrogram uses 300) because of quadratic cost. Agglomerative labels are not full-catalog assignments.
- Text similarity can reflect shared categories, countries, and generic metadata tokens; it does not imply audience preference, quality, or business performance.

**Future scope**

1. Confirm the precise catalog extraction date and use a consistent historical snapshot.
2. Conduct sensitivity analysis across field weighting, min_df, vocabulary size, SVD dimensions, and distance metrics.
3. Ask domain reviewers to assess cluster coherence and usefulness.
4. If approved, enrich with external ratings or market metadata and clearly document source/licensing; avoid turning that into rating prediction unless separately requested.
5. Use engagement data only for a separately scoped evaluation of downstream discovery/recommendation use.
""")

md("### 29. Final conclusion")
py(r'''best_k_row = k_selection.loc[k_selection.silhouette_sample.idxmax()]
tv_shares = hypotheses[0]["direction_values"]
display(Markdown(
    f"The supplied **{len(clean_df):,}-title** catalog can be represented with sparse TF-IDF and grouped into exploratory text/content segments. "
    f"The stated selection rule chose **K={selected_k}**; the best candidate silhouette in K=2–10 was {best_k_row.silhouette_sample:.3f} at K={int(best_k_row.k)}. "
    f"K-Means and sampled {model_results['selected_linkage']} agglomerative clustering provide complementary partitions; neither is ground truth. "
    f"Observed TV Show shares by addition era were {', '.join(f'{k}: {v:.1%}' for k,v in tv_shares.items())}, so the data do not support a simple monotonic increase claim. "
    "Use the clusters as reviewable catalog-segmentation hypotheses, with source-date and metadata limitations kept visible."
))''')

md("## Stage 7 — Storytelling & Video Presentation")
md(r"""### 30. Video narrative: WHY → WHAT → HOW → RESULT → BUSINESS IMPACT

Do not read code. Show the supplied catalog, selected charts, model-selection evidence, cluster examples, and limitations. The timed plan below totals 10 minutes 30 seconds, meeting the deck's requirement for a video longer than 7 minutes and within the stated approximate 20–30 minute maximum.
""")
py(r'''video_plan = pd.DataFrame([
    ("0:00–0:30", "Introduction", "Set up the question: how can a large catalog be explored through content similarity?", "Title slide"),
    ("0:30–1:00", "Business context", "Explain catalog discovery and segmentation; clarify this is not a recommendation engine.", "Business-context slide"),
    ("1:00–1:30", "Problem and objectives", "State the clustering question and why labels are unavailable.", "Problem/objectives"),
    ("1:30–2:00", "Dataset and scope", f"Report {len(clean_df):,} records, 12 observed columns, and date-added range {added.min():%Y}–{added.max():%Y}; flag the 2019-brief vs 2021-file mismatch.", "Schema and data dictionary"),
    ("2:00–2:40", "Cleaning", "Explain preserved missing metadata, zero exact duplicates, whitespace normalization, and retained unusual titles.", "Missingness/cleaning impact table"),
    ("2:40–3:45", "EDA discoveries", f"Show type mix ({movie_n:,} Movies; {tv_n:,} TV Shows), add-year and release-year trends, ratings, country/type heatmap, genres, and duration.", "EDA charts"),
    ("3:45–4:30", "Hypothesis testing", f"Explain chi-square and Mann–Whitney results. Highlight non-monotonic TV shares across eras: {tv_shares}.", "Hypothesis results table"),
    ("4:30–5:20", "Feature engineering and TF-IDF", "Describe combined fields, conservative text cleaning, TF-IDF vs counts, sparse vectors, and 100-component SVD.", "Feature flow / matrix summary"),
    ("5:20–6:20", "K-Means and K selection", f"Show elbow and silhouette evidence; explain the 1% tie rule and selected K={selected_k}.", "Elbow/silhouette charts"),
    ("6:20–7:10", "Agglomerative clustering", f"Explain Ward/average linkage sample comparison, selected {model_results['selected_linkage']} with {model_results['hierarchy_k']} groups, and sampled dendrogram.", "Dendrogram"),
    ("7:10–8:20", "Evaluation and clusters", f"Compare silhouette ({comparison.iloc[0]['silhouette_cosine']:.3f} vs {comparison.iloc[1]['silhouette_cosine']:.3f}) and show cluster sizes, terms, and representative titles.", "Metrics, cluster map, profiles"),
    ("8:20–9:10", "Business use", "Present clusters as catalog organization hypotheses and possible future recommender preprocessing; ask stakeholders to validate.", "Technical-to-business table"),
    ("9:10–9:50", "Limitations and future scope", "Mention provenance mismatch, missing fields, internal metrics, metadata-driven similarity, and sampled hierarchy.", "Limitations slide"),
    ("9:50–10:30", "Conclusion", "Restate what was learned, what clusters can support, and what evidence is needed before operational use.", "Closing slide"),
], columns=["time","section","speaking_points","visuals_to_show"])
display(video_plan)
video_plan.to_csv(TABLES / "video_presentation_plan.csv", index=False)''')

md(r"""### 31. Presentation delivery checklist

- Keep the video between **more than 7 minutes** and the deck's stated approximate **20–30 minute maximum**; this plan is about 10½ minutes.
- Explain each chart with: **what we see, why it matters, what we learned**.
- Explain each algorithm with: **why it was used, how the grouping count/linkage was selected, what segments indicate, and how a stakeholder may review them**.
- Use plain language, readable plots, and restrained business claims. The deck requests a notebook and video in a Drive folder with access for everyone; submission dates stated in the deck are 12 Oct 2026 for the two deliverables and 23 Oct 2026 for the Google Form. Confirm logistics with the course organizer.
- A separate ML submission template is mentioned in the deck but was not included with the supplied files; this notebook follows the deck's evaluation criteria and can be transferred into that template when provided.
""")

md("### 32. Output files and reproducibility")
py(r'''print("Clean data:", "data/processed/netflix_titles_cleaned.csv")
print("Figures:", "outputs/figures")
print("Tables:", "outputs/tables")
print("Models:", "outputs/models")
runtime_versions = model_results["runtime_versions"]
print("Actual runtime versions:", runtime_versions)
display(pd.DataFrame([runtime_versions], index=["version"]).T.rename_axis("package", axis=0).rename(columns={"version":"actual version"}))''')

md("### 33. Video presentation summary")
md("The presentation opens with the business need for catalog discovery, demonstrates the actual supplied-data findings, explains the TF-IDF/SVD clustering choices with measured K and internal metrics, then interprets representative segments and their limits. It ends with practical stakeholder review questions rather than unsupported performance claims.")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})
nbf.validate(nb)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
