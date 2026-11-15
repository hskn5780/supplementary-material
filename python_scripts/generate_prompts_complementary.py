"""
COMPLEMENTARY STAGE -- step 1: generate the two extra prompt configurations.

Builds the full system prompts for two configurations that were NOT part of
the original 16-run fraction of the DoE:

    code   A  B  C  D  E
    00000  0  0  0  0  0   simple prompt, no extra feature
    10010  1  0  0  1  0   sensitivity criterion + few-shot only

Factors (same order as the code digits):
    A = sensitivity criterion          (1 = present)
    B = forced chain of thought        (1 = present)
    C = prompt type                    (0 = simple, 1 = structured)
    D = few-shot examples              (1 = present)
    E = output format                  (0 = free text, 1 = structured JSON)

The text blocks are read from the 'Variáveis' sheet of DoE_TCC.xlsx and the
prompt is assembled by the SAME functions used by generate_prompts.py (they
are imported from it, not copied), so block texts, order and omissions are
identical to the original experiments. The spreadsheet and the DoE matrix
are NOT modified, and experiments_prompts.csv is NOT touched.

Output: 03_generated_prompts/complementary_prompts.csv
Columns: run_order, config_code, sensitivity, chain_of_thought, prompt_type,
         few_shot, output_format, format_tag, full_prompt
(the same columns as experiments_prompts.csv, with config_code in place of
experiment_number)

Run this from the project root, e.g.:
    python 02_scripts\\generate_prompts_complementary.py
Then REVIEW the two prompts (printed below and in the CSV -- open the CSV in
Excel read-only, never re-save it there) before running the model scripts.

Requirements: pip install openpyxl pandas
"""

import csv
from pathlib import Path

# Same folder as this script (02_scripts). Reusing the original functions
# guarantees the assembly logic is exactly the same.
from generate_prompts import DOE_SPREADSHEET_PATH, build_system_prompt, read_text_blocks

OUTPUT_CSV_PATH = "03_generated_prompts/complementary_prompts.csv"
ORIGINAL_PROMPTS_CSV_PATH = "03_generated_prompts/experiments_prompts.csv"

# Configurations to build, in the order they will be executed.
CONFIGS = ["00000", "10010"]

# Level labels, using the same vocabulary as the 'DoE' sheet. The assembly
# function only checks for the "present" labels, so the "absent" labels are
# informational (they end up in the CSV columns).
PRESENT = {
    "sensitivity": "Possuir",
    "cot": "Possui",
    "prompt_type": "Estruturado",
    "few_shot": "Possuir",
    "output_format": "Estruturado",
}
ABSENT = {
    "sensitivity": "Não possuir",
    "cot": "Não Possui",
    "prompt_type": "Simples",
    "few_shot": "Não possuir",
    "output_format": "Não estruturado",
}
FACTOR_ORDER = ["sensitivity", "cot", "prompt_type", "few_shot", "output_format"]  # A, B, C, D, E


def factors_from_code(code: str) -> dict:
    if len(code) != 5 or any(c not in "01" for c in code):
        raise ValueError(f"Invalid configuration code '{code}' (expected 5 digits of 0/1).")
    return {
        name: (PRESENT[name] if digit == "1" else ABSENT[name])
        for name, digit in zip(FACTOR_ORDER, code)
    }


def build_rows(blocks: dict) -> list[dict]:
    rows = []
    for run_order, code in enumerate(CONFIGS, start=1):
        factors = factors_from_code(code)
        full_prompt, format_tag = build_system_prompt(factors, blocks)
        rows.append(
            {
                "run_order": run_order,
                "config_code": code,
                "sensitivity": factors["sensitivity"],
                "chain_of_thought": factors["cot"],
                "prompt_type": factors["prompt_type"],
                "few_shot": factors["few_shot"],
                "output_format": factors["output_format"],
                "format_tag": format_tag,
                "full_prompt": full_prompt,
            }
        )
    return rows


def main():
    blocks = read_text_blocks(DOE_SPREADSHEET_PATH)
    rows = build_rows(blocks)

    # Sanity checks before saving.
    if len({r["full_prompt"] for r in rows}) != len(rows):
        raise ValueError("Two configurations produced identical prompts -- check the blocks.")
    original = Path(ORIGINAL_PROMPTS_CSV_PATH)
    if original.exists():
        import pandas as pd

        original_prompts = set(pd.read_csv(original)["full_prompt"])
        for r in rows:
            if r["full_prompt"] in original_prompts:
                print(f"[WARNING] Config {r['config_code']} is identical to one of the 16 original prompts.")
    else:
        print(f"[NOTE] {ORIGINAL_PROMPTS_CSV_PATH} not found; skipped the check against the original prompts.")

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"File generated: {OUTPUT_CSV_PATH} ({len(rows)} prompts)\n")
    for r in rows:
        print("=" * 70)
        print(
            f"Config {r['config_code']}  (run {r['run_order']})  format_tag={r['format_tag']}  "
            f"[{len(r['full_prompt'])} characters]"
        )
        print("=" * 70)
        print(r["full_prompt"])
        print()
    print("Review the prompts above (or the CSV, read-only) before running the model scripts.")


if __name__ == "__main__":
    main()
