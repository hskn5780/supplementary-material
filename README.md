# python_scripts

Python scripts accompanying the article **"Title and abstract screening with large language models: A factorial evaluation of prompt design and majority voting"** (Supplementary Information 3 – Python Scripts).

The scripts generate the prompts, collect title-and-abstract screening decisions from Gemini models through the Gemini Batch API, and compute the validation metrics (sensitivity, specificity, and Cohen's kappa) against the human-consensus reference standard.

## Study overview

All scripts operate on the same **290 records** (109 reference-standard INCLUDE and 181 EXCLUDE), each classified as `INCLUDE` or `EXCLUDE` from its title and abstract.

| Phase | Description | Models |
|---|---|---|
| **Phase 1** | 16-condition resolution V fractional factorial design (2^(5−1)) of five prompt factors. Pro Preview classifies each record once per condition; Flash-Lite classifies each record five times per condition, enabling single-run, best-of-three, and best-of-five comparisons. | `gemini-3.1-pro-preview`, `gemini-3.5-flash-lite` |
| **Phase 2** | Complementary comparison of the selected prompt **(ad)** against the minimal baseline **(1)**, one classification per record, with paired statistical tests. | both models |

## Scripts

### Phase 1 — Factorial design

| Order | Script | Model | What it does |
|---|---|---|---|
| 1 | `generate_prompts.py` | — | Reads the text blocks and the DoE matrix from `DoE_TCC.xlsx` and assembles the 16 full prompts in the randomized run order. Output: `experiments_prompts.csv`. |
| 2 | `run_batch_collection.py` | Pro Preview | Submits one batch of 290 requests per condition, extracts the decision (JSON or free text), and saves one CSV per condition. Resumable. |
| 3 | `calculate_validation_metrics.py` | Pro Preview | Sensitivity, specificity, PPV, NPV, accuracy, and kappa (with 95% CIs) for each of the 16 conditions. Output: `metrics_summary.csv`. |
| 4 | `run_batch_collection_flashlite_5reps.py` | Flash-Lite | Runs the 16 conditions five times (`rep_1` to `rep_5`), with `max_output_tokens = 8192`. Resumable. |
| 5 | `calculate_validation_metrics_replicates.py` | Flash-Lite | Metrics per replicate, plus the voting scenarios (best-of-three and best-of-five, sequential and fixed) with the number of API requests each would require. Outputs: `metrics_per_replicate.csv`, `metrics_voting_scenarios.csv`, `scenario_comparison.csv`. |

### Phase 2 — Complementary prompt evaluation

| Order | Script | Model | What it does |
|---|---|---|---|
| 6 | `generate_prompts_complementary.py` | — | Builds the prompts for configurations `00000` (baseline, **(1)**) and `10010` (selected prompt, **(ad)**), reusing the assembly functions of `generate_prompts.py`. Output: `complementary_prompts.csv`. |
| 7a | `run_complementary_pro.py` | Pro Preview | Runs both configurations once per record, with input validation, output validation, and resumable batch state. |
| 7b | `run_complementary_flashlite.py` | Flash-Lite | Same as 7a. |
| 8 | `calculate_validation_metrics_complementary.py` | both | Metrics for the 4 conditions (2 models × 2 prompts). Output: `metrics_summary_complementary.csv`. |
| 9 | `calculate_validation_metrics_complementary_tests.py` | both | Paired comparison **(ad) vs (1)** within each model: exact McNemar test for sensitivity (among reference-standard INCLUDE records) and specificity (among EXCLUDE records), and paired bootstrap (10,000 resamples) for the kappa difference. Output: `metrics_summary_complementary_tests.csv`. |

## Prompt factors and coding

The paper codes each factor as −1 / +1. The scripts use the equivalent **0 / 1** digits, in the order A B C D E:

| Letter | Factor | Level −1 (script 0) | Level +1 (script 1) |
|---|---|---|---|
| A | Uncertainty rule | omitted | retain records when eligibility is plausible despite incomplete information |
| B | Step-by-step reasoning | brief justification | written step-by-step evaluation before the decision |
| C | Criteria specification | simple criteria | glossary and sequential eligibility steps |
| D | Few-shot examples | omitted | four labeled examples (1 INCLUDE, 3 EXCLUDE) |
| E | Output format | free text | JSON object |

Prompt sections are assembled in the order **introductory block → C → A → B → D → E**.

Configurations in Yates notation: **(1)** = all factors at −1 (code `00000`); **(ad)** = A and D at +1 (code `10010`).

> In the spreadsheet and in the output CSVs the levels keep their original Portuguese labels (`Possuir`/`Não possuir`, `Possui`/`Não Possui`, `Estruturado`/`Simples`, `Não estruturado`), and the sensitivity column corresponds to factor A (uncertainty rule).

## Not included in these scripts

- The fractional factorial design was generated in **Minitab 22.2.1** (generator E = ABCD, defining relation I = ABCDE).
- The factorial effect analysis (Lenth's pseudo standard error, α = 0.05) and the Pareto and interaction plots were produced in **Minitab**. The design matrix and the per-condition results are in Supplementary Information 2.

## Expected folder structure

The scripts use relative paths and **must be run from the project root**, with the scripts located in `02_scripts/`:

```
project/
├── .env                              # GEMINI_API_KEY=... (do NOT commit)
├── 00_input_data/
│   └── experiment_sample/sample_290.csv
├── 01_experimental_design/
│   └── DoE_TCC.xlsx
├── 02_scripts/                       # the scripts in this folder
├── 03_generated_prompts/             # experiments_prompts.csv, complementary_prompts.csv
├── 04_experiment_results/            # data collection outputs (CSVs, raw JSONLs)
└── 05_analysis/                      # metrics tables
```

Example:

```bash
python 02_scripts/generate_prompts.py
python 02_scripts/run_batch_collection.py
python 02_scripts/calculate_validation_metrics.py
```

## Requirements

- Python 3.10 or higher
- Dependencies:

```bash
pip install pandas numpy scipy openpyxl google-genai python-dotenv
```

- For the collection scripts: a Gemini API key with billing enabled, stored in a `.env` file at the project root:

```
GEMINI_API_KEY=your_key_here
```

> **Warning:** never publish the `.env` file. Add `.env` to `.gitignore` before pushing to GitHub.

## Methodological notes

- **Labels:** AI decisions use `INCLUDE`/`EXCLUDE`; the human consensus column uses `INCLUIR`/`EXCLUIR` (Portuguese). The metrics scripts normalize both conventions (including accented variants). Unrecognized values are discarded and counted in `n_discarded`, never treated as disagreements.
- **Positive class:** `INCLUDE`.
- **Confidence intervals:** Wilson score (95%) for sensitivity and specificity; percentile bootstrap (5,000 resamples, seed 42) for kappa.
- **Voting:** replicates are evaluated in their recorded order until the required majority is reached. Because all five replicates were collected in advance, the number of requests under early stopping is estimated retrospectively. For binary labels, sequential and fixed scenarios with the same number of votes reach the same decision for each record.
- **Multiple comparisons:** McNemar p-values and the paired bootstrap interval are not adjusted for multiple comparisons.
- **Prompts:** the collection scripts do not assemble prompts; they only read the generated CSVs. Open those CSVs in Excel for reading only: saving them there converts straight quotes to curly quotes and breaks the JSON examples.
- **Configuration `00000`:** spreadsheet programs may display it as `0`. The CSV keeps the text `00000`.
- **Raw data:** original API responses are preserved in the `raw_batches/` folders (and in `batch_state/` for Phase 2), allowing decisions to be re-extracted without new API calls.
- **Model execution dates:** Gemini 3.1 Pro Preview on September 28, 2026, and Gemini 3.5 Flash-Lite on October 2, 2026, with default thinking settings.
