"""
ONE-OFF PATCH SCRIPT -- meant to be deleted after use.

Re-sends, via Batch, ONLY the rows of results_experiment_07.csv whose
ai_decision is a FORMAT_ERROR_* (truncated/unparseable response caused by an
insufficient max_output_tokens in the original run). Uses the exact same
system prompt already stored for experiment 7 in experiments_prompts.csv --
nothing about the prompt or reasoning instructions is changed, only the
generation_config.max_output_tokens fix (a request-level parameter, not part
of the prompt) is applied so the response isn't cut off again.

Updates ONLY the matching rows, in place, in the SAME
results_experiment_07.csv file -- no new file is created, row order and all
other rows are left untouched.

Requirements: pip install google-genai pandas python-dotenv
"""

import os
import re
import json
import time
from pathlib import Path

import pandas as pd
from google import genai

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NUMBER = 7  # change this if you ever need to patch another experiment
PROMPTS_CSV_PATH = Path("03_generated_prompts/experiments_prompts.csv")
RESULTS_CSV_PATH = Path(f"04_experiment_results/results_experiment_{EXPERIMENT_NUMBER:02d}.csv")
RAW_BATCHES_DIR = Path("04_experiment_results/raw_batches")

MODEL = "gemini-3.1-pro-preview"
MAX_OUTPUT_TOKENS = 8192  # same fix applied in run_batch_collection.py
POLL_INTERVAL_SECONDS = 15

# ============================================================
# DECISION EXTRACTION (same router logic as run_batch_collection.py)
# ============================================================


def extract_decision_json(response_text: str):
    text = response_text.strip()
    try:
        data = json.loads(text)
        decision = str(data.get("decision", "")).strip().upper()
        if decision in ("INCLUDE", "EXCLUDE"):
            return decision, data.get("reason", ""), data.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        pass
    block_match = re.search(r"\{.*\}", text, re.DOTALL)
    if block_match:
        try:
            data = json.loads(block_match.group(0))
            decision = str(data.get("decision", "")).strip().upper()
            if decision in ("INCLUDE", "EXCLUDE"):
                return decision, data.get("reason", ""), data.get("reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    regex_match = re.search(r'"decision"\s*:\s*"(INCLUDE|EXCLUDE)"', text, re.IGNORECASE)
    if regex_match:
        return regex_match.group(1).upper(), "[extracted via regex - invalid JSON]", ""
    return "FORMAT_ERROR_JSON", text, ""


def extract_decision_free(response_text: str):
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    if lines and lines[-1] in ("INCLUDE", "EXCLUDE"):
        return lines[-1], response_text
    occurrences = re.findall(r"\b(INCLUDE|EXCLUDE)\b", response_text)
    if occurrences:
        return occurrences[-1], response_text
    return "FORMAT_ERROR_FREE", response_text


def extract_decision(response_text: str, format_tag: str):
    if format_tag == "json":
        return extract_decision_json(response_text)
    decision, reason = extract_decision_free(response_text)
    return decision, reason, ""


# ============================================================
# STEP 1 -- find the rows that need a retry
# ============================================================


def load_failed_rows(results_path: Path) -> pd.DataFrame:
    # keep_default_na=False: empty cells (e.g. an unused ai_reasoning field)
    # stay as empty strings instead of being read as NaN/float64, which
    # would otherwise make pandas refuse to write a string back into that
    # column later.
    df = pd.read_csv(results_path, keep_default_na=False)
    failed_mask = df["ai_decision"].astype(str).str.startswith("FORMAT_ERROR")
    failed_rows = df[failed_mask]
    print(f"Loaded {len(df)} rows from {results_path}. {len(failed_rows)} need a retry.")
    return df, failed_rows


# ============================================================
# STEP 2 -- get the original prompt for this experiment
# ============================================================


def load_prompt_for_experiment(prompts_path: Path, experiment_number: int):
    prompts_df = pd.read_csv(prompts_path)
    row = prompts_df[prompts_df["experiment_number"] == experiment_number]
    if row.empty:
        raise SystemExit(f"[ERROR] experiment_number {experiment_number} not found in {prompts_path}")
    row = row.iloc[0]
    return row["full_prompt"], row["format_tag"]


# ============================================================
# STEP 3 -- submit the small batch and wait
# ============================================================


def run_retry_batch(client: genai.Client, system_prompt: str, failed_rows: pd.DataFrame) -> dict:
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RAW_BATCHES_DIR / f"retry_experiment_{EXPERIMENT_NUMBER:02d}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, article in failed_rows.iterrows():
            article_text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
            line = {
                "key": str(article["key"]),
                "request": {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": article_text}]}],
                    "generation_config": {"max_output_tokens": MAX_OUTPUT_TOKENS},
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Retry JSONL built: {jsonl_path} ({len(failed_rows)} articles)")

    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config={
            "display_name": f"retry-experiment-{EXPERIMENT_NUMBER:02d}",
            "mime_type": "jsonl",
        },
    )
    batch_job = client.batches.create(
        model=MODEL,
        src=uploaded_file.name,
        config={"display_name": f"retry-experiment-{EXPERIMENT_NUMBER:02d}"},
    )
    print(f"Retry job created: {batch_job.name} (initial state: {batch_job.state.name})")

    while batch_job.state.name not in (
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    ):
        time.sleep(POLL_INTERVAL_SECONDS)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"  ... current state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"[ERROR] retry job finished in state {batch_job.state.name}. Nothing was patched.")

    result_file_name = batch_job.dest.file_name
    content = client.files.download(file=result_file_name).decode("utf-8")

    responses_by_key = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("key")
        try:
            text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = f"[ERROR IN RAW RESPONSE]: {json.dumps(record, ensure_ascii=False)}"
        responses_by_key[key] = text

    return responses_by_key


# ============================================================
# STEP 4 -- patch ONLY the matching rows, in place, same file
# ============================================================


def patch_results_file(df: pd.DataFrame, responses_by_key: dict, format_tag: str) -> pd.DataFrame:
    still_failed = []
    for key, response_text in responses_by_key.items():
        decision, reason, reasoning = extract_decision(response_text, format_tag)
        row_mask = df["key"].astype(str) == str(key)
        df.loc[row_mask, "ai_decision"] = decision
        df.loc[row_mask, "ai_reason"] = reason
        df.loc[row_mask, "ai_reasoning"] = reasoning
        df.loc[row_mask, "raw_response"] = response_text
        if decision.startswith("FORMAT_ERROR"):
            still_failed.append(key)
        print(f"  key {key}: patched -> {decision}")

    if still_failed:
        print(f"\n[WARNING] {len(still_failed)} row(s) STILL failed after the retry: {still_failed}")
        print("These will still show up as discarded rows when you run the metrics script.")
    else:
        print("\nAll retried rows now have a valid decision.")

    return df


# ============================================================
# MAIN
# ============================================================


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (check your .env file).")

    df, failed_rows = load_failed_rows(RESULTS_CSV_PATH)
    if len(failed_rows) == 0:
        print("Nothing to retry -- no FORMAT_ERROR rows found. Exiting without changes.")
        return

    system_prompt, format_tag = load_prompt_for_experiment(PROMPTS_CSV_PATH, EXPERIMENT_NUMBER)
    print(f"Loaded original prompt for experiment {EXPERIMENT_NUMBER} (format: {format_tag}).")

    client = genai.Client(api_key=api_key)
    responses_by_key = run_retry_batch(client, system_prompt, failed_rows)

    df = patch_results_file(df, responses_by_key, format_tag)

    df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{RESULTS_CSV_PATH} updated in place ({len(responses_by_key)} row(s) patched).")


if __name__ == "__main__":
    main()
