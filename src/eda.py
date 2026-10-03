"""Exploratory summaries and matplotlib charts for the Netflix catalog."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame, figures: str | Path, tables: str | Path) -> dict:
    """Save key EDA charts/tables; all multi-valued fields are split on commas."""
    figures, tables = Path(figures), Path(tables)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    results = {}

    type_counts = df["type"].value_counts()
    type_counts.rename_axis("type").to_frame("count").to_csv(tables / "content_type_counts.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    type_counts.plot(kind="bar", color=["#d92b36", "#333333"], ax=ax)
    ax.set(title="Netflix Catalog: Movies and TV Shows", xlabel="Content type", ylabel="Titles")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, figures, "content_type_distribution.png")
    results["type_counts"] = type_counts.to_dict()

    added = pd.to_datetime(df["date_added"], errors="coerce")
    added_counts = added.dt.year.value_counts().sort_index()
    added_counts.rename_axis("year_added").to_frame("titles_added").to_csv(tables / "titles_added_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    added_counts.plot(ax=ax, marker="o", color="#d92b36")
    ax.set(title="Titles Added to Netflix by Year", xlabel="Year added", ylabel="Titles")
    _save(fig, figures, "titles_added_by_year.png")
    results["date_added_range"] = [str(added.min().date()), str(added.max().date())]

    added_type = (df.assign(year_added=added.dt.year).dropna(subset=["year_added"])
                    .groupby(["year_added", "type"]).size().unstack(fill_value=0))
    added_type.to_csv(tables / "titles_added_by_year_and_type.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    added_type.plot(ax=ax, marker="o", color=["#d92b36", "#555555"])
    ax.set(title="Movies and TV Shows Added by Year", xlabel="Year added", ylabel="Titles added")
    _save(fig, figures, "titles_added_by_year_and_type.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for content_type, group in df.groupby("type"):
        group["release_year"].plot(kind="hist", bins=30, alpha=.55, ax=ax, label=content_type)
    ax.set(title="Release Year Distribution by Content Type", xlabel="Release year", ylabel="Number of titles")
    ax.legend(title="Content type")
    _save(fig, figures, "release_year_distribution.png")

    rating_type = pd.crosstab(df["rating"].fillna("Unknown"), df["type"])
    rating_type.to_csv(tables / "rating_by_type.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    rating_type.plot(kind="bar", stacked=True, ax=ax, color=["#d92b36", "#555555"])
    ax.set(title="Netflix Content Ratings by Type", xlabel="Rating", ylabel="Titles")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, figures, "rating_distribution.png")

    def exploded_counts(column: str) -> pd.Series:
        return (df[column].dropna().str.split(", ").explode().str.strip()
                .replace("", pd.NA).dropna().value_counts())

    for col, filename, title, xlabel in [
        ("country", "country_distribution.png", "Top Countries Represented in the Catalog", "Titles with country listed"),
        ("listed_in", "genre_distribution.png", "Most Common Netflix Categories", "Titles in category"),
        ("director", "director_distribution.png", "Directors with the Most Listed Titles", "Titles"),
        ("cast", "cast_distribution.png", "Cast Members with the Most Listed Titles", "Titles"),
    ]:
        counts = exploded_counts(col).head(15)
        counts.rename_axis(col).to_frame("title_mentions").to_csv(tables / f"top_{col}.csv")
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.sort_values().plot(kind="barh", ax=ax, color="#d92b36")
        ax.set(title=title, xlabel=xlabel, ylabel=col.replace("_", " ").title())
        _save(fig, figures, filename)

    country_type = (df.assign(country=df["country"].fillna("").str.split(", "))
                      .explode("country").query("country != ''")
                      .groupby(["country", "type"]).size().unstack(fill_value=0))
    top_countries = country_type.sum(axis=1).nlargest(15).index
    country_type.loc[top_countries].to_csv(tables / "country_by_type_top15.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    country_type.loc[top_countries].sort_values("Movie").plot(kind="barh", stacked=True, ax=ax,
                                                                  color=["#d92b36", "#555555"])
    ax.set(title="Content Type by Top Country", xlabel="Titles listing country", ylabel="Country")
    _save(fig, figures, "country_by_content_type.png")

    proportions = country_type.loc[top_countries].div(country_type.loc[top_countries].sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(proportions, annot=True, fmt=".0%", cmap="Reds", linewidths=.4, ax=ax)
    ax.set(title="Content-Type Mix Across Top Countries", xlabel="Content type", ylabel="Country")
    _save(fig, figures, "country_type_heatmap.png")

    # Duration is stored as text and mixes movie minutes with TV seasons.
    duration = df["duration"].astype("string")
    df_duration = df.assign(duration_value=pd.to_numeric(duration.str.extract(r"(\d+)")[0], errors="coerce"))
    movie_duration = df_duration.loc[df_duration["type"].eq("Movie"), "duration_value"].dropna()
    tv_duration = df_duration.loc[df_duration["type"].eq("TV Show"), "duration_value"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(movie_duration, bins=30, color="#d92b36", edgecolor="white")
    ax.set(title="Movie Duration Distribution", xlabel="Duration (minutes)", ylabel="Number of movies")
    _save(fig, figures, "movie_duration_minutes.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    season_counts = tv_duration.value_counts().sort_index()
    season_counts.plot(kind="bar", ax=ax, color="#333333")
    ax.set(title="TV Show Duration Distribution", xlabel="Duration (seasons)", ylabel="Number of TV shows")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, figures, "tv_show_duration_seasons.png")
    # Remove a previous run's misleading shared-scale figure, if it exists.
    (figures / "duration_by_type.png").unlink(missing_ok=True)
    results["duration_medians"] = {"Movie_minutes": float(movie_duration.median()),
                                   "TV_Show_seasons": float(tv_duration.median())}

    year_type = pd.crosstab(df["release_year"], df["type"])
    year_type.to_csv(tables / "release_year_by_type.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    year_type.loc[year_type.index >= 2000].plot(ax=ax, color=["#d92b36", "#555555"])
    ax.set(title="Release-Year Trend Since 2000", xlabel="Release year", ylabel="Titles in supplied catalog")
    _save(fig, figures, "release_year_by_content_type.png")
    return results
