"""
Generates the master file with the 16 fully assembled prompts, built from
DoE_TCC.xlsx, in the randomized run order defined for data collection.

Run this script ONCE. Review the resulting CSV (you may open it in Excel for
READ-ONLY inspection -- never edit and re-save it there: Excel's autocorrect
replaces straight quotes with curly quotes and corrupts the JSON few-shot
examples).

Output: experiments_prompts.csv
Columns: run_order, experiment_number, sensitivity, chain_of_thought,
         prompt_type, few_shot, output_format, format_tag, full_prompt
"""

import csv
import openpyxl

DOE_SPREADSHEET_PATH = "01_experimental_design/DoE_TCC.xlsx"
OUTPUT_CSV_PATH = "03_generated_prompts/experiments_prompts.csv"


def read_text_blocks(spreadsheet_path: str) -> dict:
    """Reads each text block by its exact position on the 'Variáveis' sheet.
    Note: the sheet itself (tab name and cell values) stays in Portuguese,
    since it is the original source of truth maintained outside this script."""
    wb = openpyxl.load_workbook(spreadsheet_path, data_only=True)
    ws = wb["Variáveis"]

    def cell(row, col):
        return ws.cell(row=row, column=col).value

    blocks = {
        "block0": cell(2, 4),
        "var3_structured": cell(3, 4),
        "var3_simple": cell(3, 5),
        "var1_present": cell(4, 4),
        "var1_absent": cell(4, 5),
        "var2_present": cell(5, 4),
        "var2_absent": cell(5, 5),
        "var4_absent": cell(6, 5),
        "var4_json_with_reasoning": cell(6, 4),
        "var4_json_without_reasoning": cell(7, 4),
        "var4_free_with_reasoning": cell(8, 4),
        "var4_free_without_reasoning": cell(9, 4),
        "var5_structured_with_reasoning": cell(10, 4),
        "var5_structured_without_reasoning": cell(11, 4),
        "var5_free": cell(10, 5),
    }

    missing = [k for k, v in blocks.items() if not v]
    if missing:
        raise ValueError(f"Empty blocks read from the spreadsheet (check the layout): {missing}")
    return blocks


def read_experiments(spreadsheet_path: str) -> list[dict]:
    """Reads the 'DoE' sheet directly. Each row already carries both:
    OrdemPad (the experiment's canonical/standard-order identifier) and
    OrdemEns (the run order, 1 to 16). Returns a list of dicts, one per
    experiment, sorted by run_order -- no external run-order list needed,
    the spreadsheet is the single source of truth for both."""
    wb = openpyxl.load_workbook(spreadsheet_path, data_only=True)
    ws = wb["DoE"]

    experiments = []
    for row in ws.iter_rows(min_row=2, max_row=17, max_col=9, values_only=True):
        ordem_pad, ordem_ens, _, _, sensitivity, cot, prompt_type, few_shot, output_format = row
        experiments.append(
            {
                "experiment_number": int(ordem_pad),  # OrdemPad
                "run_order": int(ordem_ens),           # OrdemEns
                "sensitivity": sensitivity,
                "cot": cot,
                "prompt_type": prompt_type,
                "few_shot": few_shot,
                "output_format": output_format,
            }
        )

    ids = [e["run_order"] for e in experiments]
    if sorted(ids) != list(range(1, len(experiments) + 1)):
        raise ValueError(
            f"OrdemEns values are not a clean 1..{len(experiments)} sequence: {sorted(ids)}"
        )

    experiments.sort(key=lambda e: e["run_order"])
    return experiments


def build_system_prompt(factors: dict, blocks: dict):
    """Assembles the system prompt for one experiment, following the fixed
    hierarchy: Block0 -> Var3 -> Var1 -> Var2 -> Var4 -> Var5.
    Returns (prompt_text, format_tag) where format_tag is 'json' or 'free'."""

    parts = [blocks["block0"]]

    if factors["prompt_type"] == "Estruturado":
        parts.append(blocks["var3_structured"])
    else:
        parts.append(blocks["var3_simple"])

    if factors["sensitivity"] == "Possuir":
        parts.append(blocks["var1_present"])

    cot_present = factors["cot"] == "Possui"
    parts.append(blocks["var2_present"] if cot_present else blocks["var2_absent"])

    structured_output = factors["output_format"] == "Estruturado"
    format_tag = "json" if structured_output else "free"

    if factors["few_shot"] == "Possuir":
        if structured_output:
            var4_block = blocks["var4_json_with_reasoning"] if cot_present else blocks["var4_json_without_reasoning"]
        else:
            var4_block = blocks["var4_free_with_reasoning"] if cot_present else blocks["var4_free_without_reasoning"]
        parts.append(var4_block)

    var5_block = (
        (blocks["var5_structured_with_reasoning"] if cot_present else blocks["var5_structured_without_reasoning"])
        if structured_output
        else blocks["var5_free"]
    )
    parts.append(var5_block)

    return "\n\n".join(parts), format_tag


def main():
    blocks = read_text_blocks(DOE_SPREADSHEET_PATH)
    experiments = read_experiments(DOE_SPREADSHEET_PATH)

    output_rows = []
    for exp in experiments:
        full_prompt, format_tag = build_system_prompt(exp, blocks)
        output_rows.append(
            {
                "run_order": exp["run_order"],
                "experiment_number": exp["experiment_number"],
                "sensitivity": exp["sensitivity"],
                "chain_of_thought": exp["cot"],
                "prompt_type": exp["prompt_type"],
                "few_shot": exp["few_shot"],
                "output_format": exp["output_format"],
                "format_tag": format_tag,
                "full_prompt": full_prompt,
            }
        )

    fields = list(output_rows[0].keys())
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"File generated: {OUTPUT_CSV_PATH} ({len(output_rows)} prompts)")
    print("Run order (run_order -> experiment_number):")
    for row in output_rows:
        print(f"  {row['run_order']:>2} -> experiment {row['experiment_number']}")
    print("\nReview the CSV (read-only) before running the actual data collection.")


if __name__ == "__main__":
    main()
