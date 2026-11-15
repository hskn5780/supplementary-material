"""
Calculates Sensitivity, Specificity, and Cohen's Kappa for each of the 16
DoE experiments, comparing the AI decision ('ai_decision' column, produced
by run_batch_collection.py) against the human gold standard/consensus
('human_consensus' column, carried over from the original article sample).

Reads every results_experiment_NN.csv file in 04_experiment_results/, cross
references each one with 03_generated_prompts/experiments_prompts.csv to
attach the factor levels for that experiment, and writes one consolidated
summary row per experiment to 05_analysis/metrics_summary.csv.

Requirements: pip install pandas numpy
"""

import re
import unicodedata
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================
# CONFIGURATION
# =================================================================
RESULTS_DIR = Path("04_experiment_results")
PROMPTS_CSV_PATH = Path("03_generated_prompts/experiments_prompts.csv")
OUTPUT_DIR = Path("05_analysis")
OUTPUT_CSV_PATH = OUTPUT_DIR / "metrics_summary.csv"

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


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Reads a CSV tolerating encodings other than plain UTF-8 (common in
    exports made on Excel/Windows)."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


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


def compute_metrics_for_experiment(results_path: Path) -> dict:
    """Reads one results_experiment_NN.csv and returns a dict with all
    metrics for that experiment. Returns an 'error' key instead if the file
    is missing, empty, or has no valid labeled rows."""
    try:
        df = read_csv_with_fallback(results_path)
    except FileNotFoundError:
        return {"error": f"file not found: {results_path}"}
    except pd.errors.EmptyDataError:
        return {"error": f"file is empty: {results_path}"}

    for col in (AI_COLUMN, HUMAN_COLUMN):
        if col not in df.columns:
            return {"error": f"column '{col}' not found. Columns present: {list(df.columns)}"}

    if len(df) == 0:
        return {"error": "file has no data rows"}

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

    return {
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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not PROMPTS_CSV_PATH.exists():
        raise SystemExit(
            f"[ERROR] '{PROMPTS_CSV_PATH}' not found. Run generate_prompts.py first, "
            f"and run this script from the project root."
        )
    prompts_df = pd.read_csv(PROMPTS_CSV_PATH).sort_values("run_order")

    summary_rows = []
    for _, prompt_row in prompts_df.iterrows():
        experiment_number = int(prompt_row["experiment_number"])
        run_order = int(prompt_row["run_order"])
        results_path = RESULTS_DIR / f"results_experiment_{experiment_number:02d}.csv"

        metrics = compute_metrics_for_experiment(results_path)

        row = {
            "run_order": run_order,
            "experiment_number": experiment_number,
            "sensitivity_factor": prompt_row["sensitivity"],
            "chain_of_thought": prompt_row["chain_of_thought"],
            "prompt_type": prompt_row["prompt_type"],
            "few_shot": prompt_row["few_shot"],
            "output_format": prompt_row["output_format"],
        }
        row.update(metrics)
        summary_rows.append(row)

        if "error" in metrics:
            print(f"Experiment {experiment_number:>2} (run {run_order:>2}): [SKIPPED] {metrics['error']}")
        else:
            print(
                f"Experiment {experiment_number:>2} (run {run_order:>2}): "
                f"n={metrics['n_evaluated']:>3}  "
                f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                f"kappa={metrics['kappa']:.3f} ({metrics['kappa_interpretation']})"
                + (f"  [{metrics['n_discarded']} row(s) discarded]" if metrics["n_discarded"] else "")
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
