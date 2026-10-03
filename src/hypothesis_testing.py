"""Pre-specified, assumption-checked tests grounded in the inspected data."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def run_hypothesis_tests(df: pd.DataFrame, alpha=0.05):
    """Test catalog associations; output details, assumptions, and caveats."""
    rows = []

    # H1 tests association across broad date-added eras, then uses proportions
    # to determine direction. This is not a causal test of Netflix strategy.
    added = pd.to_datetime(df["date_added"], errors="coerce")
    eras = pd.cut(added.dt.year, bins=[2007, 2014, 2017, 2021],
                  labels=["2008-2014", "2015-2017", "2018-2021"])
    valid = df["type"].notna() & eras.notna()
    tab = pd.crosstab(df.loc[valid, "type"], eras.loc[valid])
    chi2, p, dof, expected = chi2_contingency(tab, correction=False)
    shares = (tab.loc["TV Show"] / tab.sum(axis=0)).to_dict()
    n_h1 = int(tab.to_numpy().sum())
    cramers_v_h1 = float(np.sqrt(chi2 / (n_h1 * min(tab.shape[0] - 1, tab.shape[1] - 1))))
    rows.append({"hypothesis": "H1: Catalog content type is associated with date-added era",
                 "null": "Content type and date-added era are independent",
                 "test": "Pearson chi-square test of independence",
                 "variables": "type x date_added era (3 pre-defined broad periods)",
                 "statistic": float(chi2), "degrees_of_freedom": int(dof), "p_value": float(p),
                 "alpha": alpha, "reject_null": bool(p < alpha),
                 "assumption_min_expected": float(np.min(expected)),
                 "n_observations": n_h1, "cramers_v": cramers_v_h1,
                 "observed_table": tab.to_dict(), "effect_direction": "TV Show share by era",
                 "direction_values": {str(k): float(v) for k, v in shares.items()},
                 "interpretation": "Association indicates the type mix differs across eras; compare observed TV shares for direction. Date-added records describe the supplied catalog, not Netflix's full historical commissioning or availability."})

    # H2 uses one primary (first-listed) country per title, avoiding repeated
    # title counts from exploding multi-country metadata.
    country = df["country"].fillna("").str.split(",").str[0].str.strip()
    country = country.where(country.ne(""), pd.NA)
    top = country.value_counts().head(10).index
    country_group = country.where(country.isin(top), "Other")
    valid = country_group.notna() & df["type"].notna()
    ctab = pd.crosstab(country_group[valid], df.loc[valid, "type"])
    chi2c, pc, dofc, expectedc = chi2_contingency(ctab, correction=False)
    n_h2 = int(ctab.to_numpy().sum())
    cramers_v_h2 = float(np.sqrt(chi2c / (n_h2 * min(ctab.shape[0] - 1, ctab.shape[1] - 1))))
    rows.append({"hypothesis": "H2: Content type distribution differs by primary listed country",
                 "null": "Content type and primary listed country group are independent",
                 "test": "Pearson chi-square test of independence",
                 "variables": "primary country (top 10 plus Other) x type",
                 "statistic": float(chi2c), "degrees_of_freedom": int(dofc), "p_value": float(pc),
                 "alpha": alpha, "reject_null": bool(pc < alpha),
                 "assumption_min_expected": float(np.min(expectedc)),
                 "n_observations": n_h2, "cramers_v": cramers_v_h2,
                 "observed_table": ctab.to_dict(),
                 "interpretation": "A significant association is descriptive, not causal. Country metadata may be incomplete; the first-listed country is a consistent proxy and does not represent all co-production countries."})

    # Release year is discrete and skewed; Mann-Whitney compares distributions
    # without assuming normality. The dataset includes one row per title.
    movie_years = df.loc[df["type"].eq("Movie"), "release_year"].dropna()
    tv_years = df.loc[df["type"].eq("TV Show"), "release_year"].dropna()
    u, pm = mannwhitneyu(movie_years, tv_years, alternative="two-sided", method="auto")
    rank_biserial = float(2 * u / (len(movie_years) * len(tv_years)) - 1)
    rows.append({"hypothesis": "H3: Release-year distributions differ between Movies and TV Shows",
                 "null": "Release-year distributions are equal across content types",
                 "test": "Two-sided Mann-Whitney U test",
                 "variables": "release_year by type; independent title rows",
                 "statistic": float(u), "degrees_of_freedom": None, "p_value": float(pm),
                 "alpha": alpha, "reject_null": bool(pm < alpha),
                 "assumption_min_expected": None,
                 "median_movie_year": float(movie_years.median()),
                 "median_tv_year": float(tv_years.median()),
                 "n_movie": int(len(movie_years)), "n_tv_show": int(len(tv_years)),
                 "rank_biserial_movie_vs_tv": rank_biserial,
                 "interpretation": "A significant result supports a difference in release-year distributions; it does not imply content type caused release timing or measure business performance."})

    # Preserve raw p-values and add Holm family-wise adjustment across the
    # three planned tests. This does not replace the unadjusted results.
    p_values = np.asarray([row["p_value"] for row in rows], dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    n_tests = len(p_values)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (n_tests - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    for row, adj_p in zip(rows, adjusted):
        row["p_value_holm"] = float(adj_p)
        row["reject_null_holm"] = bool(adj_p < alpha)
    return rows
