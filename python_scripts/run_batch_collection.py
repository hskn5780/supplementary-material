"""
Runs the 16 DoE experiments, one at a time, in the order defined by
experiments_prompts.csv (generated beforehand by generate_prompts.py).

This script does NOT assemble any prompt -- it only READS ready-made prompts
from the master CSV and submits them. This separates the "assemble and
review the prompts" step from the "spend tokens running" step, reducing the
risk of errors during the actual data collection.

For each row of the master CSV (one experiment), in 'run_order':
  1. Builds a JSONL file with one request per article (always the same 290).
  2. Uploads the file, creates the batch job, waits for it to finish.
  3. Downloads the result, extracts the decision (routing between the JSON
     parser and the free-text parser, based on that experiment's
     'format_tag' column).
  4. Saves one result CSV per experiment.

Requirements:
    pip install google-genai pandas python-dotenv
    Create a .env file at the project root with: GEMINI_API_KEY=your_paid_key
"""

import os
import re
import json
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# ============================================================
# CONFIGURATION
# ============================================================

PROMPTS_CSV_PATH = "03_generated_prompts/experiments_prompts.csv"
ARTICLES_CSV_PATH = "00_input_data/experiment_sample/sample_290.csv"
OUTPUT_DIR = Path("04_experiment_results")
RAW_BATCHES_DIR = OUTPUT_DIR / "raw_batches"
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_BATCHES_DIR.mkdir(exist_ok=True)

MODEL = "gemini-3.1-pro-preview"
POLL_INTERVAL_SECONDS = 30
QUOTA_RETRY_SECONDS = 30 * 60  # wait 30 minutes when the quota is exhausted (HTTP 429)

# ============================================================
# DECISION EXTRACTION (JSON / free-text router)
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
# BATCH JSONL BUILDING
# ============================================================


def build_jsonl(system_prompt: str, articles_df: pd.DataFrame, jsonl_path: Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, article in articles_df.iterrows():
            article_text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
            line = {
                "key": str(article["key"]),
                "request": {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": article_text}]}],
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ============================================================
# RUN ONE FULL EXPERIMENT
# ============================================================


def run_experiment(client: genai.Client, prompt_row: pd.Series, articles_df: pd.DataFrame) -> None:
    experiment_number = int(prompt_row["experiment_number"])
    run_order = int(prompt_row["run_order"])
    system_prompt = prompt_row["full_prompt"]
    format_tag = prompt_row["format_tag"]

    # Resume support: if this experiment's result CSV already exists, skip it.
    # (The CSV is only written at the very end of a successful run, so its
    # existence means the experiment is complete.)
    existing_output = OUTPUT_DIR / f"results_experiment_{experiment_number:02d}.csv"
    if existing_output.exists():
        print(
            f"\nRun {run_order}/16 -- Experiment number {experiment_number}: "
            f"result already exists ({existing_output}). Skipping."
        )
        return

    print(
        f"\n{'='*60}\n"
        f"Run {run_order}/16 -- Experiment number {experiment_number} "
        f"(format: {format_tag})\n{'='*60}"
    )

    jsonl_path = RAW_BATCHES_DIR / f"batch_experiment_{experiment_number:02d}.jsonl"
    build_jsonl(system_prompt, articles_df, jsonl_path)
    print(f"  JSONL built: {jsonl_path} ({len(articles_df)} articles)")

    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config={
            "display_name": f"batch-experiment-{experiment_number:02d}",
            "mime_type": "jsonl",
        },
    )
    print(f"  File uploaded: {uploaded_file.name}")

    while True:
        try:
            batch_job = client.batches.create(
                model=MODEL,
                src=uploaded_file.name,
                config={"display_name": f"screening-experiment-{experiment_number:02d}"},
            )
            break
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                print(
                    f"  Quota exhausted (429). Trying again in "
                    f"{QUOTA_RETRY_SECONDS // 60} minutes "
                    f"({time.strftime('%H:%M:%S')} now)..."
                )
                time.sleep(QUOTA_RETRY_SECONDS)
            else:
                raise
    print(f"  Job created: {batch_job.name} (initial state: {batch_job.state.name})")

    while batch_job.state.name not in (
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    ):
        time.sleep(POLL_INTERVAL_SECONDS)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"  ... current state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  ERROR: job finished in state {batch_job.state.name}. Skipping to the next one.")
        return

    result_file_name = batch_job.dest.file_name
    content_bytes = client.files.download(file=result_file_name)
    content = content_bytes.decode("utf-8")

    responses_by_key = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("key")
        try:
            response_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            response_text = f"[ERROR IN RAW RESPONSE]: {json.dumps(record, ensure_ascii=False)}"
        responses_by_key[key] = response_text

    result_rows = []
    for _, article in articles_df.iterrows():
        key = str(article["key"])
        response_text = responses_by_key.get(key, "[NO RESPONSE - KEY NOT FOUND IN RESULT]")
        decision, reason, reasoning = extract_decision(response_text, format_tag)
        result_rows.append(
            {
                "key": key,
                "title": article["title"],
                "abstract": article["abstract"],
                "human_consensus": article.get("Gabarito_Humano", ""),
                "ai_decision": decision,
                "ai_reason": reason,
                "ai_reasoning": reasoning,
                "raw_response": response_text,
            }
        )

    result_df = pd.DataFrame(result_rows)
    output_path = OUTPUT_DIR / f"results_experiment_{experiment_number:02d}.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    n_errors = result_df["ai_decision"].str.startswith("FORMAT_ERROR").sum()
    print(f"  Done. Results saved to: {output_path}")
    print(f"  Formatting errors: {n_errors} out of {len(result_df)} articles")


# ============================================================
# MAIN
# ============================================================


def main():
    # Loads .env from the project root, if present (pip install python-dotenv).
    # If the package isn't installed or the file doesn't exist, falls back to
    # a plain system environment variable.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set -- create a .env file at the project root "
            "with the line GEMINI_API_KEY=your_key_here, or export the environment variable."
        )

    client = genai.Client(api_key=api_key)

    prompts_df = pd.read_csv(PROMPTS_CSV_PATH).sort_values("run_order")
    articles_df = pd.read_csv(ARTICLES_CSV_PATH)

    print(f"{len(prompts_df)} experiments loaded from {PROMPTS_CSV_PATH}.")
    print(f"{len(articles_df)} articles per experiment (from {ARTICLES_CSV_PATH}).")
    print("Run order (experiment_number):", prompts_df["experiment_number"].tolist())

    for _, prompt_row in prompts_df.iterrows():
        run_experiment(client, prompt_row, articles_df)

    print("\nAll experiments have been processed.")


if __name__ == "__main__":
    main()
