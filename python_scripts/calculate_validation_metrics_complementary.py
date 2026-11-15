"""
COMPLEMENTARY STAGE -- metrics.

Calculates Sensitivity, Specificity, and Cohen's Kappa for the 4 complementary
conditions (2 models x 2 prompt configurations, ONE vote per article), comparing
the AI decision ('ai_decision' column, produced by run_complementary_pro.py /
run_complementary_flashlite.py) against the human gold standard/consensus
('human_consensus' column, carried over from the original article sample).

Same calculations as calculate_validation_metrics.py (the helper functions
below are unchanged), applied to:

    04_experiment_results/complementary/<MODEL>/results_config_<code>.csv

Each file is cross-referenced with 03_generated_prompts/complementary_prompts.csv
to attach the factor levels of its configuration, and one consolidated summary
row per condition is written to:

    05_analysis/complementary/metrics_summary_complementary.csv

The columns are the same as in metrics_summary.csv (with 'model' and
'config_code' in place of 'experiment_number'), so the rows can be stacked with
the single-run results of the original experiments for comparison.

Integrity checks added to the original logic (a file failing them is
SKIPPED and reported, never silently calculated): 'model' and 'config_code'
columns must match the folder/file the data came from, and article keys must
be unique. A row count different from EXPECTED_N_ARTICLES only triggers a
warning (n_evaluated shows how many rows were actually used).

Requirements: pip install pandas numpy
"""

import unicodedata
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================
# CONFIGURATION
# =================================================================
MODELS = ["gemini-3.1-pro-preview", "gemini-3.5-flash-lite"]   # folder names inside the results folder
RESULTS_DIR = Path("04_experiment_results") / "complementary"
PROMPTS_CSV_PATH = Path("03_generated_prompts/complementary_prompts.csv")
OUTPUT_DIR = Path("05_analysis") / "complementary"
OUTPUT_CSV_PATH = OUTPUT_DIR / "metrics_summary_complementary.csv"

EXPECTED_N_ARTICLES = 290
AI_COLUMN = "ai_decision"
HUMAN_COLUMN = "human_consensus"
POSITIVE_CLASS = "INCLUDE"       # label treated as "positive" for sensitivity/specificity
N_BOOTSTRAP = 5000                # number of resamples for the Kappa CI
SEED = 42                         # fixed seed -> reproducible CI across runs
# =================================================================

# The two label columns may use different languages/spellings for the same
# category (ai_decision: INCLUDE/EXCLUDE; human_consensus: INCLUIR/EXCLUIR,
# in Portuguese, since it comes from the original article sample). Left
# unhandled, "INCLUIR" != "INCLUDE" would make the script count EVERY row as
# a disagreement, even when the AI and the human actually agree, silently
# producing a badly wrong Kappa. This map normalizes either convention (and
# accented variants) to a single standard.
LABEL_MAP = {
    "INCLUDE": "INCLUDE", "INCLUIR": "INCLUDE", "INCLUIDO": "INCLUDE", "INCLUIDA": "INCLUDE",
    "EXCLUDE": "EXCLUDE", "EXCLUIR": "EXCLUDE", "EXCLUIDO": "EXCLUDE", "EXCLUIDA": "EXCLUDE",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Uppercase, strip whitespace, strip accents, then map PT/EN spellings
    to a single label. Values not found in LABEL_MAP become NaN (treated as
    invalid and discarded/reported later -- never silently counted as a
    disagreement)."""
    unaccented = series.astype(str).str.strip().str.upper().map(
        lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    )
    return unaccented.map(LABEL_MAP)


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Reads a CSV tolerating encodings other than plain UTF-8 (common in
    exports made on Excel/Windows)."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def confusion_matrix_2x2(y_true, y_pred, labels):
    """
    Builds a 2x2 confusion matrix by hand (no scikit-learn dependency).
    labels = [positive_class, negative_class].
    Returns a 2x2 array: row/col 0 = positive_class, 1 = negative_class
    (same convention as sklearn.metrics.confusion_matrix(labels=labels)).
    """
    pos, neg = labels
    matrix = np.zeros((2, 2), dtype=int)
    index = {pos: 0, neg: 1}
    for true, pred in zip(y_true, y_pred):
        matrix[index[true], index[pred]] += 1
    return matrix


def cohen_kappa(y_true, y_pred, labels):
    """
    Cohen's Kappa using the standard formula (Cohen, 1960):
        kappa = (Po - Pe) / (1 - Pe)
    Po = observed agreement; Pe = agreement expected by chance, given each
    rater's marginal distribution.
    """
    matrix = confusion_matrix_2x2(y_true, y_pred, labels)
    n = matrix.sum()
    if n == 0:
        return float("nan")
    po = np.trace(matrix) / n
    true_margin = matrix.sum(axis=1) / n
    pred_margin = matrix.sum(axis=0) / n
    pe = np.sum(true_margin * pred_margin)
    if pe == 1:
        return 1.0 if po == 1 else float("nan")
    return (po - pe) / (1 - pe)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) scale, the most commonly cited reference."""
    if k < 0:
        return "poor agreement (< 0)"
    if k <= 0.20:
        return "slight agreement"
    if k <= 0.40:
        return "fair agreement"
    if k <= 0.60:
        return "moderate agreement"
    if k <= 0.80:
        return "substantial agreement"
    return "almost perfect agreement"


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval (1927) for a proportion -- more reliable than
    the plain normal-approximation CI when the proportion is close to 0 or
    1 (common in sensitivity/specificity)."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (center - margin) / denom, (center + margin) / denom


def kappa_bootstrap_ci(y_human, y_ai, labels, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% CI for Kappa via bootstrap (resampling rows with replacement).
    Avoids relying on Kappa's asymptotic standard-error formula, which is a
    weaker approximation on smaller samples."""
    rng = np.random.default_rng(seed)
    y_human = np.asarray(y_human)
    y_ai = np.asarray(y_ai)
    n = len(y_human)
    kappas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        kappas[i] = cohen_kappa(y_human[idx], y_ai[idx], labels=labels)
    return np.percentile(kappas, 2.5), np.percentile(kappas, 97.5)


def compute_metrics_for_condition(results_path: Path, model: str, config_code: str, incomplete_path: Path = None) -> dict:
    """Reads one results_config_<code>.csv and returns a dict with all
    metrics for that condition (same metrics as the original script). Returns
    an 'error' key instead if the file is missing, empty, inconsistent, or has
    no valid labeled rows. A 'warning' key may be added to a normal result."""
    try:
        # dtype=str keeps config_code as text ("00000", not 0)
        df = read_csv_with_fallback(results_path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        extra = ""
        if incomplete_path is not None and incomplete_path.exists():
            extra = (f" ({incomplete_path.name} exists: the run did not pass validation, "
                     f"see batch_state/validation_report_config_{config_code}.json)")
        return {"error": f"file not found: {results_path}{extra}"}
    except pd.errors.EmptyDataError:
        return {"error": f"file is empty: {results_path}"}

    for col in (AI_COLUMN, HUMAN_COLUMN, "key", "model", "config_code"):
        if col not in df.columns:
            return {"error": f"column '{col}' not found. Columns present: {list(df.columns)}"}

    if len(df) == 0:
        return {"error": "file has no data rows"}

    # Integrity: the data must belong to the model/configuration it is filed under.
    if (df["model"] != model).any():
        return {"error": f"'model' column contains {sorted(set(df['model']))}, expected only '{model}'"}
    if (df["config_code"] != config_code).any():
        return {"error": f"'config_code' column contains {sorted(set(df['config_code']))}, expected only '{config_code}'"}
    n_duplicated = int(df["key"].duplicated().sum())
    if n_duplicated:
        return {"error": f"{n_duplicated} duplicated article key(s); each article must appear exactly once"}
    warning = ""
    if len(df) != EXPECTED_N_ARTICLES:
        warning = f"file has {len(df)} rows, expected {EXPECTED_N_ARTICLES}"

    df["_AI_NORM"] = normalize_labels(df[AI_COLUMN])
    df["_HUMAN_NORM"] = normalize_labels(df[HUMAN_COLUMN])

    valid = df["_AI_NORM"].notna() & df["_HUMAN_NORM"].notna()
    n_discarded = len(df) - valid.sum()
    discarded_examples = ""
    if n_discarded:
        examples = df.loc[~valid, [AI_COLUMN, HUMAN_COLUMN]].drop_duplicates().head(10)
        discarded_examples = examples.to_string(index=False).replace("\n", " | ")

    df_valid = df[valid]
    if len(df_valid) == 0:
        return {"error": "no rows left after label normalization"}

    y_human = df_valid["_HUMAN_NORM"]
    y_ai = df_valid["_AI_NORM"]

    negative_class = "EXCLUDE" if POSITIVE_CLASS == "INCLUDE" else "INCLUDE"
    labels = [POSITIVE_CLASS, negative_class]

    matrix = confusion_matrix_2x2(y_human, y_ai, labels=labels)
    tp, fn = matrix[0, 0], matrix[0, 1]
    fp, tn = matrix[1, 0], matrix[1, 1]
    n = tp + fn + fp + tn

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    kappa = cohen_kappa(y_human, y_ai, labels=labels)

    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    kappa_ci = kappa_bootstrap_ci(y_human, y_ai, labels)

    result = {
        "n_evaluated": n,
        "n_discarded": n_discarded,
        "discarded_examples": discarded_examples,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sensitivity, "sensitivity_ci_low": sens_ci[0], "sensitivity_ci_high": sens_ci[1],
        "specificity": specificity, "specificity_ci_low": spec_ci[0], "specificity_ci_high": spec_ci[1],
        "ppv": ppv, "npv": npv, "accuracy": accuracy,
        "kappa": kappa, "kappa_ci_low": kappa_ci[0], "kappa_ci_high": kappa_ci[1],
        "kappa_interpretation": interpret_kappa(kappa) if not np.isnan(kappa) else "n/a",
    }
    if warning:
        result["warning"] = warning
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPTS_CSV_PATH.exists():
        raise SystemExit(
            f"[ERROR] '{PROMPTS_CSV_PATH}' not found. Run generate_prompts_complementary.py first, "
            f"and run this script from the project root."
        )
    # dtype=str: config_code must stay text ("00000" would otherwise become 0)
    prompts_df = pd.read_csv(PROMPTS_CSV_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    prompts_df["run_order"] = prompts_df["run_order"].astype(int)
    prompts_df = prompts_df.sort_values("run_order")

    summary_rows = []
    for model in MODELS:
        for _, prompt_row in prompts_df.iterrows():
            config_code = str(prompt_row["config_code"])
            run_order = int(prompt_row["run_order"])
            results_path = RESULTS_DIR / model / f"results_config_{config_code}.csv"
            incomplete_path = RESULTS_DIR / model / f"incomplete_config_{config_code}.csv"

            metrics = compute_metrics_for_condition(results_path, model, config_code, incomplete_path)

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
            summary_rows.append(row)

            label = f"{model} | config {config_code}"
            if "error" in metrics:
                print(f"{label}: [SKIPPED] {metrics['error']}")
            else:
                print(
                    f"{label}: "
                    f"n={metrics['n_evaluated']:>3}  "
                    f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                    f"kappa={metrics['kappa']:.3f} ({metrics['kappa_interpretation']})"
                    + (f"  [{metrics['n_discarded']} row(s) discarded]" if metrics["n_discarded"] else "")
                    + (f"  [WARNING: {metrics['warning']}]" if "warning" in metrics else "")
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to: {OUTPUT_CSV_PATH}")
    print("Note: spreadsheet programs may display config_code '00000' as 0; the CSV itself keeps the text.")


if __name__ == "__main__":
    main()
