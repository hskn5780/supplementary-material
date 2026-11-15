"""
COMPLEMENTARY STAGE -- metrics + paired tests (config 10010 vs config 00000).

Produces ONE file with everything in metrics_summary_complementary.csv (same
4 rows -- 2 models x 2 configurations -- and the same columns, calculated by
the very same functions of calculate_validation_metrics_complementary.py,
which is imported, not copied) PLUS extra 'cmp_' columns with the paired
comparison of the two configurations within each model:

  1. Sensitivity: exact McNemar test (two-sided binomial test on the
     discordant pairs), using only the articles the humans marked INCLUDE.
  2. Specificity: exact McNemar test, using only the articles the humans
     marked EXCLUDE.
  3. Kappa: paired bootstrap of the difference (kappa 10010 - kappa 00000).
     Each resample draws the SAME articles for the human, the 00000 and the
     10010 classifications; the 95% CI is the percentile interval of the
     differences.

Both configurations classified the same articles, so the tests are paired:
the comparison is made article by article, not between two independent
intervals.

Reading the 'cmp_' columns
--------------------------
* The comparison belongs to the PAIR (one per model). Its values are repeated
  on the two rows of that model (00000 and 10010) for convenience -- count
  each pair once if you aggregate.
* All differences are 10010 minus 00000. 'favors_10010' (b) counts articles
  that 10010 got right and 00000 got wrong; 'favors_00000' (c) is the
  opposite. Only these discordant articles enter the McNemar test.
* The McNemar p-values and the bootstrap interval are NOT adjusted for
  multiple comparisons (2 models x 3 tests).

Safety: the paired comparison is only calculated if both files exist, have
unique keys and exactly EXPECTED_N_ARTICLES rows, contain only valid labels,
cover the same articles and carry the same human consensus. Otherwise the
'cmp_error' column explains why (nothing is dropped or guessed), and the
regular metrics are still written.

Output: 05_analysis/complementary/metrics_summary_complementary_tests.csv

Requirements: pip install pandas numpy scipy
(this script must stay in the same folder as
calculate_validation_metrics_complementary.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import calculate_validation_metrics_complementary as base
except ImportError:
    raise SystemExit(
        "[ERROR] Could not import calculate_validation_metrics_complementary.py. Keep this script in the same "
        "folder (02_scripts) and run it from the project root, e.g.:\n"
        "    python 02_scripts\\calculate_validation_metrics_complementary_tests.py"
    )

from scipy.stats import binomtest

# =================================================================
# CONFIGURATION (models, folders, label handling and the 290-article check
# are shared with calculate_validation_metrics_complementary.py)
# =================================================================
BASE_CONFIG = "00000"             # reference configuration
TEST_CONFIG = "10010"             # configuration compared against the reference
N_BOOTSTRAP_PAIRED = 10000        # resamples for the kappa-difference interval
SEED_PAIRED = 42                  # fixed seed -> reproducible interval
OUTPUT_CSV_PATH = base.OUTPUT_DIR / "metrics_summary_complementary_tests.csv"

POS, NEG = "INCLUDE", "EXCLUDE"
LABELS = [base.POSITIVE_CLASS, "EXCLUDE" if base.POSITIVE_CLASS == "INCLUDE" else "INCLUDE"]

COMPARISON_COLUMNS = [
    "cmp_pair",
    "cmp_n_paired",
    "cmp_sens_n_relevant",
    "cmp_sens_diff",
    f"cmp_sens_favors_{TEST_CONFIG}",
    f"cmp_sens_favors_{BASE_CONFIG}",
    "cmp_sens_n_discordant",
    "cmp_sens_mcnemar_p",
    "cmp_spec_n_negative",
    "cmp_spec_diff",
    f"cmp_spec_favors_{TEST_CONFIG}",
    f"cmp_spec_favors_{BASE_CONFIG}",
    "cmp_spec_n_discordant",
    "cmp_spec_mcnemar_p",
    "cmp_kappa_diff",
    "cmp_kappa_diff_ci_low",
    "cmp_kappa_diff_ci_high",
    "cmp_kappa_n_valid_resamples",
    "cmp_kappa_conclusion",
    "cmp_error",
]


class PairingError(Exception):
    """The two configurations cannot be safely paired."""


# =================================================================
# PAIRING
# =================================================================


def load_paired(model: str) -> pd.DataFrame:
    """Loads the results of both configurations of one model and returns one
    row per article: key, human, ai_<BASE>, ai_<TEST> (labels normalized to
    INCLUDE/EXCLUDE). Raises PairingError at the first integrity problem."""
    frames = {}
    for code in (BASE_CONFIG, TEST_CONFIG):
        path = base.RESULTS_DIR / model / f"results_config_{code}.csv"
        try:
            df = base.read_csv_with_fallback(path, dtype=str, keep_default_na=False)
        except FileNotFoundError:
            raise PairingError(f"file not found: {path}")
        except pd.errors.EmptyDataError:
            raise PairingError(f"file is empty: {path}")

        missing = [c for c in ("key", "model", "config_code", base.HUMAN_COLUMN, base.AI_COLUMN) if c not in df.columns]
        if missing:
            raise PairingError(f"{path.name}: missing column(s) {missing}")
        if (df["model"] != model).any() or (df["config_code"] != code).any():
            raise PairingError(f"{path.name}: 'model'/'config_code' columns do not match {model} / {code}")
        if df["key"].duplicated().any():
            raise PairingError(f"{path.name}: {int(df['key'].duplicated().sum())} duplicated article key(s)")
        if len(df) != base.EXPECTED_N_ARTICLES:
            raise PairingError(f"{path.name}: {len(df)} rows, expected {base.EXPECTED_N_ARTICLES}")

        human = base.normalize_labels(df[base.HUMAN_COLUMN])
        ai = base.normalize_labels(df[base.AI_COLUMN])
        for name, series, raw in (("human_consensus", human, df[base.HUMAN_COLUMN]), ("ai_decision", ai, df[base.AI_COLUMN])):
            if series.isna().any():
                examples = sorted(set(raw[series.isna()]))[:5]
                raise PairingError(f"{path.name}: {int(series.isna().sum())} invalid label(s) in {name}, e.g. {examples}")
        frames[code] = pd.DataFrame({"key": df["key"], "human": human, "ai": ai})

    a, b = frames[BASE_CONFIG], frames[TEST_CONFIG]
    if set(a["key"]) != set(b["key"]):
        raise PairingError("the two files do not contain the same articles")
    merged = a.merge(b, on="key", suffixes=(f"_{BASE_CONFIG}", f"_{TEST_CONFIG}"), validate="one_to_one")
    if len(merged) != base.EXPECTED_N_ARTICLES:
        raise PairingError(f"{len(merged)} articles after pairing, expected {base.EXPECTED_N_ARTICLES}")
    if (merged[f"human_{BASE_CONFIG}"] != merged[f"human_{TEST_CONFIG}"]).any():
        n = int((merged[f"human_{BASE_CONFIG}"] != merged[f"human_{TEST_CONFIG}"]).sum())
        raise PairingError(f"human_consensus differs between the two files for {n} article(s)")
    merged = merged.rename(columns={f"human_{BASE_CONFIG}": "human"}).drop(columns=[f"human_{TEST_CONFIG}"])
    return merged.rename(columns={f"ai_{BASE_CONFIG}": "ai_base", f"ai_{TEST_CONFIG}": "ai_test"})


# =================================================================
# TESTS
# =================================================================


def mcnemar_exact(correct_base: np.ndarray, correct_test: np.ndarray):
    """Exact two-sided McNemar test. correct_* are boolean arrays (is the
    article classified correctly?). Returns (b, c, n_discordant, p) where
    b = test right & base wrong, c = test wrong & base right. With no
    discordant pairs, p = 1."""
    b = int(np.sum(correct_test & ~correct_base))
    c = int(np.sum(~correct_test & correct_base))
    n_disc = b + c
    p = binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue if n_disc > 0 else 1.0
    return b, c, n_disc, float(p)


def fast_kappa(h: np.ndarray, p: np.ndarray) -> float:
    """Cohen's kappa for boolean arrays (True = INCLUDE). Same formula as
    base.cohen_kappa, vectorised for the bootstrap; NaN when undefined."""
    n = h.size
    tp = np.count_nonzero(h & p)
    fn = np.count_nonzero(h & ~p)
    fp = np.count_nonzero(~h & p)
    tn = n - tp - fn - fp
    po = (tp + tn) / n
    pe = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
    if pe == 1:
        return float("nan")
    return (po - pe) / (1 - pe)


def paired_kappa_bootstrap(h, p0, p1, n_boot=N_BOOTSTRAP_PAIRED, seed=SEED_PAIRED):
    """Paired bootstrap of kappa(p1) - kappa(p0). Returns (ci_low, ci_high,
    n_valid_resamples). Resamples where either kappa is undefined are
    discarded and counted."""
    rng = np.random.default_rng(seed)
    n = h.size
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)          # the SAME articles for the three classifications
        hb = h[idx]
        k0 = fast_kappa(hb, p0[idx])
        k1 = fast_kappa(hb, p1[idx])
        if np.isfinite(k0) and np.isfinite(k1):
            diffs.append(k1 - k0)
    diffs = np.array(diffs)
    if diffs.size == 0:
        raise PairingError("no valid bootstrap resample (kappa undefined in all of them)")
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high), int(diffs.size)


def compare_configs(model: str) -> dict:
    """All 'cmp_' columns for one model (paired comparison TEST vs BASE)."""
    out = {col: np.nan for col in COMPARISON_COLUMNS}
    out["cmp_pair"] = f"{TEST_CONFIG} vs {BASE_CONFIG}"
    try:
        df = load_paired(model)
    except PairingError as e:
        out["cmp_error"] = str(e)
        return out
    out["cmp_error"] = ""

    human_pos = (df["human"] == POS).to_numpy()
    p0 = (df["ai_base"] == POS).to_numpy()
    p1 = (df["ai_test"] == POS).to_numpy()
    out["cmp_n_paired"] = len(df)

    # ---- sensitivity: articles the humans marked INCLUDE; correct = INCLUDE ----
    if human_pos.any():
        b, c, n_disc, p = mcnemar_exact(p0[human_pos], p1[human_pos])
        out.update({
            "cmp_sens_n_relevant": int(human_pos.sum()),
            "cmp_sens_diff": float(p1[human_pos].mean() - p0[human_pos].mean()),
            f"cmp_sens_favors_{TEST_CONFIG}": b,
            f"cmp_sens_favors_{BASE_CONFIG}": c,
            "cmp_sens_n_discordant": n_disc,
            "cmp_sens_mcnemar_p": p,
        })
    else:
        out["cmp_error"] = "no article with human_consensus INCLUDE: sensitivity test not possible"

    # ---- specificity: articles the humans marked EXCLUDE; correct = EXCLUDE ----
    human_neg = ~human_pos
    if human_neg.any():
        b, c, n_disc, p = mcnemar_exact(~p0[human_neg], ~p1[human_neg])
        out.update({
            "cmp_spec_n_negative": int(human_neg.sum()),
            "cmp_spec_diff": float((~p1[human_neg]).mean() - (~p0[human_neg]).mean()),
            f"cmp_spec_favors_{TEST_CONFIG}": b,
            f"cmp_spec_favors_{BASE_CONFIG}": c,
            "cmp_spec_n_discordant": n_disc,
            "cmp_spec_mcnemar_p": p,
        })
    else:
        out["cmp_error"] = "no article with human_consensus EXCLUDE: specificity test not possible"

    # ---- kappa: observed difference (same function as the summary) + paired bootstrap ----
    k0 = base.cohen_kappa(df["human"], df["ai_base"], labels=LABELS)
    k1 = base.cohen_kappa(df["human"], df["ai_test"], labels=LABELS)
    out["cmp_kappa_diff"] = float(k1 - k0)
    try:
        low, high, n_valid = paired_kappa_bootstrap(human_pos, p0, p1)
    except PairingError as e:
        out["cmp_error"] = str(e)
        return out
    out["cmp_kappa_diff_ci_low"], out["cmp_kappa_diff_ci_high"], out["cmp_kappa_n_valid_resamples"] = low, high, n_valid
    if low > 0:
        out["cmp_kappa_conclusion"] = f"95% CI entirely positive: favors {TEST_CONFIG}"
    elif high < 0:
        out["cmp_kappa_conclusion"] = f"95% CI entirely negative: favors {BASE_CONFIG}"
    else:
        out["cmp_kappa_conclusion"] = "95% CI includes zero: inconclusive"
    return out


def print_comparison(model: str, cmp: dict) -> None:
    print(f"\n--- {model}: {cmp['cmp_pair']} ---")
    if cmp["cmp_error"]:
        print(f"  [COMPARISON ERROR] {cmp['cmp_error']}")
        if np.isnan(cmp["cmp_sens_mcnemar_p"]):
            return
    t, b = TEST_CONFIG, BASE_CONFIG
    print(f"  Paired articles: {cmp['cmp_n_paired']}")
    print(
        f"  Sensitivity (n={cmp['cmp_sens_n_relevant']}): difference {cmp['cmp_sens_diff'] * 100:+.2f} pp | "
        f"favors {t}: {cmp[f'cmp_sens_favors_{t}']} | favors {b}: {cmp[f'cmp_sens_favors_{b}']} | "
        f"McNemar exact p = {cmp['cmp_sens_mcnemar_p']:.5f}"
    )
    print(
        f"  Specificity (n={cmp['cmp_spec_n_negative']}): difference {cmp['cmp_spec_diff'] * 100:+.2f} pp | "
        f"favors {t}: {cmp[f'cmp_spec_favors_{t}']} | favors {b}: {cmp[f'cmp_spec_favors_{b}']} | "
        f"McNemar exact p = {cmp['cmp_spec_mcnemar_p']:.5f}"
    )
    if not np.isnan(cmp["cmp_kappa_diff_ci_low"]):
        print(
            f"  Kappa difference: {cmp['cmp_kappa_diff']:+.4f} | 95% bootstrap CI "
            f"[{cmp['cmp_kappa_diff_ci_low']:+.4f}; {cmp['cmp_kappa_diff_ci_high']:+.4f}] "
            f"({cmp['cmp_kappa_n_valid_resamples']} valid resamples) -> {cmp['cmp_kappa_conclusion']}"
        )


# =================================================================
# MAIN
# =================================================================


def main():
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not base.PROMPTS_CSV_PATH.exists():
        raise SystemExit(
            f"[ERROR] '{base.PROMPTS_CSV_PATH}' not found. Run generate_prompts_complementary.py first, "
            f"and run this script from the project root."
        )
    prompts_df = pd.read_csv(base.PROMPTS_CSV_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    prompts_df["run_order"] = prompts_df["run_order"].astype(int)
    prompts_df = prompts_df.sort_values("run_order")
    codes = set(prompts_df["config_code"])
    if not {BASE_CONFIG, TEST_CONFIG} <= codes:
        raise SystemExit(f"[ERROR] {base.PROMPTS_CSV_PATH} must contain configurations {BASE_CONFIG} and {TEST_CONFIG}; found {sorted(codes)}")

    summary_rows = []
    for model in base.MODELS:
        comparison = compare_configs(model)       # once per model; repeated on both rows of the pair
        for _, prompt_row in prompts_df.iterrows():
            config_code = str(prompt_row["config_code"])
            run_order = int(prompt_row["run_order"])
            results_path = base.RESULTS_DIR / model / f"results_config_{config_code}.csv"
            incomplete_path = base.RESULTS_DIR / model / f"incomplete_config_{config_code}.csv"

            metrics = base.compute_metrics_for_condition(results_path, model, config_code, incomplete_path)

            row = {
                "model": model,
                "run_order": run_order,
                "config_code": config_code,
                "sensitivity_factor": prompt_row["sensitivity"],
                "chain_of_thought": prompt_row["chain_of_thought"],
                "prompt_type": prompt_row["prompt_type"],
                "few_shot": prompt_row["few_shot"],
                "output_format": prompt_row["output_format"],
            }
            row.update(metrics)
            row.update(comparison)
            summary_rows.append(row)

            label = f"{model} | config {config_code}"
            if "error" in metrics:
                print(f"{label}: [SKIPPED] {metrics['error']}")
            else:
                print(
                    f"{label}: n={metrics['n_evaluated']:>3}  "
                    f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                    f"kappa={metrics['kappa']:.3f} ({metrics['kappa_interpretation']})"
                    + (f"  [{metrics['n_discarded']} row(s) discarded]" if metrics["n_discarded"] else "")
                    + (f"  [WARNING: {metrics['warning']}]" if "warning" in metrics else "")
                )
        print_comparison(model, comparison)

    summary_df = pd.DataFrame(summary_rows)
    # metrics columns first (same order as metrics_summary_complementary.csv), 'cmp_' columns last
    ordered = [c for c in summary_df.columns if not c.startswith("cmp_")] + COMPARISON_COLUMNS
    summary_df = summary_df[ordered]
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to: {OUTPUT_CSV_PATH}")
    print("Notes: the 'cmp_' values belong to the pair and are repeated on both rows of each model; "
          "p-values and intervals are not adjusted for multiple comparisons.")


if __name__ == "__main__":
    main()
