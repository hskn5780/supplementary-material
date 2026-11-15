"""
Calculates Sensitivity, Specificity, and Cohen's Kappa for the 16 DoE
experiments of a REPLICATED study (N_REPLICATES runs of every experiment),
comparing the AI decision ('ai_decision' column, produced by
run_batch_collection*.py) against the human gold standard/consensus
('human_consensus' column, carried over from the original article sample).

It produces three outputs, all in 05_analysis/<MODEL>/:

  1. metrics_per_replicate.csv
        The same metrics as the original script, but one row per
        (experiment, replicate) -- i.e. what you would get by running each
        replicate on its own (a single call per article).

  2. metrics_voting_scenarios.csv
        One row per (experiment, voting scenario), using the votes of the
        replicates for each article:
          - best_of_3_sequential : run 2 votes; if they agree, stop (2 calls);
                                   otherwise run a 3rd vote that decides.
          - best_of_5_sequential : run 3 votes; if all agree, stop (3 calls);
                                   otherwise keep adding votes until one label
                                   reaches 3 votes (4 or 5 calls).
          - fixed_3_votes        : always run 3 votes; majority decides.
          - fixed_5_votes        : always run 5 votes; majority decides.
        Besides the usual metrics, each row has the number of API calls the
        scenario would have consumed (total, average per article, and how they
        were distributed).

  3. scenario_comparison.csv
        One row per scenario, averaged across the 16 experiments, so you can
        see what is worth it: accuracy vs number of calls (cost).

Important note: for binary labels, "sequential" and "fixed" scenarios with the
same number of votes reach the SAME decision for every article (stopping early
does not change the majority). So best_of_3_sequential and fixed_3_votes have
identical metrics, and so do best_of_5_sequential and fixed_5_votes; they
differ only in how many calls they need.

Invalid answers (FORMAT_ERROR_*) never count as a vote for either label. If an
article ends without a clear majority (e.g. a tie caused by an invalid vote),
it is left out of the metrics and reported in n_discarded -- never silently
counted as a disagreement.

Expected input layout (created by run_batch_collection_flashlite_5reps.py):
    04_experiment_results/<MODEL>/rep_<n>/results_experiment_NN.csv

Requirements: pip install pandas numpy
"""

import re
import unicodedata
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================
# CONFIGURATION
# =================================================================
MODEL = "gemini-3.5-flash-lite"   # must match the folder name inside 04_experiment_results/
N_REPLICATES = 5

RESULTS_DIR = Path("04_experiment_results") / MODEL
PROMPTS_CSV_PATH = Path("03_generated_prompts/experiments_prompts.csv")
OUTPUT_DIR = Path("05_analysis") / MODEL
PER_REPLICATE_CSV_PATH = OUTPUT_DIR / "metrics_per_replicate.csv"
VOTING_CSV_PATH = OUTPUT_DIR / "metrics_voting_scenarios.csv"
COMPARISON_CSV_PATH = OUTPUT_DIR / "scenario_comparison.csv"

KEY_COLUMN = "key"
AI_COLUMN = "ai_decision"
HUMAN_COLUMN = "human_consensus"
POSITIVE_CLASS = "INCLUDE"       # label treated as "positive" for sensitivity/specificity
N_BOOTSTRAP = 5000                # number of resamples for the Kappa CI
SEED = 42                         # fixed seed -> reproducible CI across runs

# Voting scenarios.
#   kind="sequential": votes are taken in replicate order (rep_1, rep_2, ...)
#       and counting stops as soon as one label reaches `threshold` votes.
#   kind="fixed": all `n_reps` votes are always used; the majority decides.
SCENARIOS = {
    "best_of_3_sequential": {"kind": "sequential", "n_reps": 3, "threshold": 2},
    "best_of_5_sequential": {"kind": "sequential", "n_reps": 5, "threshold": 3},
    "fixed_3_votes": {"kind": "fixed", "n_reps": 3},
    "fixed_5_votes": {"kind": "fixed", "n_reps": 5},
}
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


def compute_metrics_from_labels(y_human, y_ai, n_discarded=0, discarded_examples="") -> dict:
    """Computes all metrics from two aligned series of already-normalized,
    valid labels ('INCLUDE'/'EXCLUDE'). Shared by the per-replicate analysis
    and the voting scenarios, so both use exactly the same formulas."""
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

    return compute_metrics_from_labels(
        df_valid["_HUMAN_NORM"], df_valid["_AI_NORM"], n_discarded, discarded_examples
    )


# =================================================================
# VOTING
# =================================================================


def majority_decision(counts: dict):
    """Label with strictly more valid votes; NaN on a tie or with no valid votes."""
    if counts["INCLUDE"] > counts["EXCLUDE"]:
        return "INCLUDE"
    if counts["EXCLUDE"] > counts["INCLUDE"]:
        return "EXCLUDE"
    return np.nan


def sequential_vote(votes, threshold: int):
    """Counts votes in order and stops as soon as one label reaches
    `threshold` valid votes. Returns (decision, number_of_calls_used).
    Invalid votes (NaN) still cost a call but count for neither label. If no
    label ever reaches the threshold, all votes are used and the majority of
    the valid ones decides (NaN on a tie)."""
    counts = {"INCLUDE": 0, "EXCLUDE": 0}
    for i, vote in enumerate(votes, start=1):
        if vote in counts:
            counts[vote] += 1
            if counts[vote] >= threshold:
                return vote, i
    return majority_decision(counts), len(votes)


def fixed_vote(votes):
    """Always uses every vote; majority of the valid ones decides."""
    counts = {"INCLUDE": 0, "EXCLUDE": 0}
    for vote in votes:
        if vote in counts:
            counts[vote] += 1
    return majority_decision(counts), len(votes)


def results_path_for(experiment_number: int, replicate: int) -> Path:
    return RESULTS_DIR / f"rep_{replicate}" / f"results_experiment_{experiment_number:02d}.csv"


def load_votes_for_experiment(experiment_number: int):
    """Returns (votes_df, problems). votes_df is indexed by article key with a
    'human' column and one 'rep_N' column (normalized AI label) per replicate
    that could be loaded. 'problems' lists replicates that were not usable."""
    labels = {}
    human = None
    problems = []

    for rep in range(1, N_REPLICATES + 1):
        path = results_path_for(experiment_number, rep)
        if not path.exists():
            problems.append(f"rep_{rep}: file not found")
            continue
        try:
            df = read_csv_with_fallback(path)
        except pd.errors.EmptyDataError:
            problems.append(f"rep_{rep}: file is empty")
            continue

        missing_cols = [c for c in (KEY_COLUMN, AI_COLUMN, HUMAN_COLUMN) if c not in df.columns]
        if missing_cols:
            problems.append(f"rep_{rep}: missing column(s) {missing_cols}")
            continue

        df[KEY_COLUMN] = df[KEY_COLUMN].astype(str)
        if df[KEY_COLUMN].duplicated().any():
            problems.append(f"rep_{rep}: duplicated article keys")
            continue

        keys = df[KEY_COLUMN].values
        labels[f"rep_{rep}"] = pd.Series(normalize_labels(df[AI_COLUMN]).values, index=keys)
        if human is None:
            human = pd.Series(normalize_labels(df[HUMAN_COLUMN]).values, index=keys)

    if not labels:
        return None, problems

    votes_df = pd.DataFrame(labels)
    votes_df.insert(0, "human", human)
    return votes_df, problems


def evaluate_scenario(votes_df: pd.DataFrame, scenario: dict) -> dict:
    """Applies one voting scenario to every article and returns the metrics
    plus the number of API calls it would have used."""
    n_reps = scenario["n_reps"]
    rep_cols = [f"rep_{r}" for r in range(1, n_reps + 1)]
    unavailable = [c for c in rep_cols if c not in votes_df.columns]
    if unavailable:
        return {"error": f"replicate(s) not available: {unavailable}"}

    vote_matrix = votes_df[rep_cols].to_numpy(dtype=object)
    decisions, calls = [], []
    for votes in vote_matrix:
        if scenario["kind"] == "sequential":
            decision, n_calls = sequential_vote(list(votes), scenario["threshold"])
        else:
            decision, n_calls = fixed_vote(list(votes))
        decisions.append(decision)
        calls.append(n_calls)

    decision_series = pd.Series(decisions, index=votes_df.index, dtype=object)
    valid = decision_series.notna() & votes_df["human"].notna()
    n_discarded = int((~valid).sum())

    discarded_examples = ""
    if n_discarded:
        examples = []
        for key in votes_df.index[~valid][:10]:
            shown = "/".join("invalid" if pd.isna(v) else str(v) for v in votes_df.loc[key, rep_cols])
            examples.append(f"{key}: {shown}")
        discarded_examples = " | ".join(examples)

    if valid.sum() == 0:
        return {"error": "no articles left after voting/label normalization"}

    metrics = compute_metrics_from_labels(
        votes_df.loc[valid, "human"], decision_series[valid], n_discarded, discarded_examples
    )

    distribution = Counter(calls)
    metrics.update({
        "n_reps_max": n_reps,
        "total_calls": int(sum(calls)),
        "avg_calls_per_article": float(np.mean(calls)),
        "calls_distribution": "|".join(f"{k}:{distribution[k]}" for k in sorted(distribution)),
    })
    return metrics


# =================================================================
# COMPARISON ACROSS SCENARIOS
# =================================================================


def build_scenario_comparison(per_rep_df: pd.DataFrame, voting_df: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario, averaged across experiments. 'single_run' is the
    baseline: each replicate analysed alone (1 call per article)."""
    rows = []

    def summarize(name, df, calls_col, max_calls):
        valid = df[df["kappa"].notna()]
        if valid.empty:
            return
        avg_calls = float(valid[calls_col].mean()) if calls_col else 1.0
        rows.append({
            "scenario": name,
            "n_experiments_or_runs": len(valid),
            "mean_sensitivity": valid["sensitivity"].mean(),
            "min_sensitivity": valid["sensitivity"].min(),
            "mean_specificity": valid["specificity"].mean(),
            "min_specificity": valid["specificity"].min(),
            "mean_kappa": valid["kappa"].mean(),
            "min_kappa": valid["kappa"].min(),
            "mean_calls_per_article": avg_calls,
            "max_calls_per_article": max_calls,
            "calls_saved_vs_max_pct": 100 * (1 - avg_calls / max_calls),
        })

    if "kappa" in per_rep_df.columns:
        summarize("single_run (each replicate alone)", per_rep_df, None, 1)
    if "kappa" in voting_df.columns:
        for name, scenario in SCENARIOS.items():
            subset = voting_df[voting_df["scenario"] == name]
            if len(subset):
                summarize(name, subset, "avg_calls_per_article", scenario["n_reps"])

    return pd.DataFrame(rows)


# =================================================================
# MAIN
# =================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPTS_CSV_PATH.exists():
        raise SystemExit(
            f"[ERROR] '{PROMPTS_CSV_PATH}' not found. Run generate_prompts.py first, "
            f"and run this script from the project root."
        )
    if not RESULTS_DIR.exists():
        raise SystemExit(
            f"[ERROR] '{RESULTS_DIR}' not found. Check MODEL at the top of this script and "
            f"run it from the project root."
        )
    prompts_df = pd.read_csv(PROMPTS_CSV_PATH).sort_values("run_order")

    per_rep_rows = []
    voting_rows = []

    for _, prompt_row in prompts_df.iterrows():
        experiment_number = int(prompt_row["experiment_number"])
        run_order = int(prompt_row["run_order"])

        factors = {
            "run_order": run_order,
            "experiment_number": experiment_number,
            "sensitivity_factor": prompt_row["sensitivity"],
            "chain_of_thought": prompt_row["chain_of_thought"],
            "prompt_type": prompt_row["prompt_type"],
            "few_shot": prompt_row["few_shot"],
            "output_format": prompt_row["output_format"],
        }

        print(f"\nExperiment {experiment_number:>2} (run {run_order:>2})")

        # ---- 1) each replicate on its own ----
        for rep in range(1, N_REPLICATES + 1):
            metrics = compute_metrics_for_experiment(results_path_for(experiment_number, rep))
            per_rep_rows.append({**factors, "replicate": rep, **metrics})

            if "error" in metrics:
                print(f"  rep {rep}: [SKIPPED] {metrics['error']}")
            else:
                print(
                    f"  rep {rep}: n={metrics['n_evaluated']:>3}  "
                    f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                    f"kappa={metrics['kappa']:.3f} ({metrics['kappa_interpretation']})"
                    + (f"  [{metrics['n_discarded']} row(s) discarded]" if metrics["n_discarded"] else "")
                )

        # ---- 2) voting scenarios ----
        votes_df, problems = load_votes_for_experiment(experiment_number)
        for scenario_name, scenario in SCENARIOS.items():
            if votes_df is None:
                metrics = {"error": "no replicate could be loaded: " + "; ".join(problems)}
            else:
                metrics = evaluate_scenario(votes_df, scenario)
            voting_rows.append({**factors, "scenario": scenario_name, **metrics})

            if "error" in metrics:
                print(f"  {scenario_name:<22} [SKIPPED] {metrics['error']}")
            else:
                saved = 100 * (1 - metrics["avg_calls_per_article"] / metrics["n_reps_max"])
                print(
                    f"  {scenario_name:<22} n={metrics['n_evaluated']:>3}  "
                    f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
                    f"kappa={metrics['kappa']:.3f}  "
                    f"calls/article={metrics['avg_calls_per_article']:.2f} "
                    f"(saves {saved:.0f}% vs {metrics['n_reps_max']})"
                    + (f"  [{metrics['n_discarded']} row(s) discarded]" if metrics["n_discarded"] else "")
                )

    per_rep_df = pd.DataFrame(per_rep_rows)
    voting_df = pd.DataFrame(voting_rows)
    comparison_df = build_scenario_comparison(per_rep_df, voting_df)

    per_rep_df.to_csv(PER_REPLICATE_CSV_PATH, index=False, encoding="utf-8-sig")
    voting_df.to_csv(VOTING_CSV_PATH, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON (averages across experiments)")
    print("=" * 70)
    if len(comparison_df):
        print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("No results available to compare yet.")

    print(f"\nPer-replicate metrics saved to: {PER_REPLICATE_CSV_PATH}")
    print(f"Voting-scenario metrics saved to: {VOTING_CSV_PATH}")
    print(f"Scenario comparison saved to: {COMPARISON_CSV_PATH}")


if __name__ == "__main__":
    main()
