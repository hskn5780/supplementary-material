"""
COMPLEMENTARY STAGE -- step 3: run the two extra configurations (00000 and
10010) on Gemini Flash Lite.

Same procedure as run_batch_collection.py: reads the ready-made prompts from
03_generated_prompts/complementary_prompts.csv (created by
generate_prompts_complementary.py), and for each configuration builds a JSONL
with one request per article (the same 290 articles), submits it through the
Gemini Batch API, waits for it, extracts the decisions and saves the results.

ONE call per article per configuration (a single vote): no repetitions and no
majority voting. 2 configurations x 290 articles = 580 classifications.

Safeguards added on top of the original procedure
-------------------------------------------------
1. Inputs are validated BEFORE any API call: prompts CSV (config codes,
   format tags, required fields, code vs. factor columns) and the articles CSV
   (required columns, 290 rows, unique/non-empty keys, valid human labels).
2. A configuration is only "complete" when its 290 decisions pass validation:
   exactly one row per article, no missing/duplicated/unexpected keys, no
   API errors, no incomplete (truncated) responses, no FORMAT_ERROR, and
   every decision is INCLUDE or EXCLUDE. An existing CSV is re-validated on
   every run; an invalid one is NOT treated as success.
3. The JSONL returned by the API is saved byte for byte before anything is
   parsed (errors and metadata preserved).
4. Batch identifiers (uploaded file, batch name, state) are recorded in a
   state file, so an interrupted run resumes the SAME batch instead of
   sending a new one.
5. Free-text parser: the decision is the LAST non-empty line of the response
   and it must be exactly INCLUDE or EXCLUDE (or "EXCLUDED", read as
   EXCLUDE). Words found inside the
   justification are never used (a truncated answer becomes FORMAT_ERROR).
6. The model ID is checked against the API (models.get, no cost) at start.

Files (MODEL = the model ID below)
----------------------------------
04_experiment_results/complementary/<MODEL>/
    results_config_<code>.csv              final, validated results (only written when valid)
    incomplete_config_<code>.csv           all rows, written instead when validation fails
    raw_batches/batch_config_<code>.jsonl      requests sent
    raw_batches/api_result_config_<code>.jsonl exact JSONL returned by the API
    batch_state/batch_state_config_<code>.json batch identifiers / progress
    batch_state/validation_report_config_<code>.json  problems found (if any)

Nothing is ever overwritten silently: older files are renamed '*.prev_<time>*'.

Requirements:
    pip install google-genai pandas python-dotenv
    .env file at the project root with: GEMINI_API_KEY=your_paid_key
"""

import os
import re
import sys
import json
import time
import hashlib
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# ============================================================
# CONFIGURATION
# ============================================================

PROMPTS_CSV_PATH = "03_generated_prompts/complementary_prompts.csv"
ARTICLES_CSV_PATH = "00_input_data/experiment_sample/sample_290.csv"

EXPECTED_CONFIGS = ["00000", "10010"]
EXPECTED_N_ARTICLES = 290

MODEL = "gemini-3.5-flash-lite"
RESULTS_DIR = Path("04_experiment_results") / "complementary" / MODEL
RAW_BATCHES_DIR = RESULTS_DIR / "raw_batches"
STATE_DIR = RESULTS_DIR / "batch_state"

POLL_INTERVAL_SECONDS = 30
QUOTA_RETRY_SECONDS = 30 * 60  # wait 30 minutes when the quota is exhausted (HTTP 429)

# Shows the model, the plan and the number of calls, and waits for you to
# type YES before spending anything. Set to False to run unattended.
REQUIRE_CONFIRMATION = True

LABELS = ("INCLUDE", "EXCLUDE")
TERMINAL_STATES = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
)
FAILED_STATES = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")

# Same vocabulary accepted by the metrics script for the human gold standard.
HUMAN_LABEL_MAP = {
    "INCLUDE": "INCLUDE", "INCLUIR": "INCLUDE", "INCLUIDO": "INCLUDE", "INCLUIDA": "INCLUDE",
    "EXCLUDE": "EXCLUDE", "EXCLUIR": "EXCLUDE", "EXCLUIDO": "EXCLUDE", "EXCLUIDA": "EXCLUDE",
}

# ============================================================
# SMALL HELPERS (hashing, timestamps, safe writes)
# ============================================================


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def archive(path: Path) -> Path:
    """Renames an existing file to '<name>.prev_<time><ext>' so nothing is lost."""
    new_path = path.with_name(f"{path.stem}.prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    n = 1
    while new_path.exists():
        new_path = path.with_name(f"{path.stem}.prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{n}{path.suffix}")
        n += 1
    path.rename(new_path)
    return new_path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def as_text(value) -> str:
    return "" if pd.isna(value) else str(value)


def normalize_human_label(value):
    text = unicodedata.normalize("NFKD", as_text(value).strip().upper()).encode("ascii", "ignore").decode("utf-8")
    return HUMAN_LABEL_MAP.get(text)


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
    """Strict: the decision is the LAST non-empty line, and it must be exactly
    INCLUDE or EXCLUDE (the single variant "EXCLUDED" is also accepted and
    normalised to EXCLUDE). INCLUDE/EXCLUDE words inside the justification are
    never extracted, so an incomplete answer is reported as FORMAT_ERROR_FREE
    instead of being guessed."""
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]

    if lines and lines[-1] in LABELS:
        return lines[-1], response_text

    # Unambiguous variant of exclusion
    if lines and lines[-1] == "EXCLUDED":
        return "EXCLUDE", response_text

    return "FORMAT_ERROR_FREE", response_text


def extract_decision(response_text: str, format_tag: str):
    if format_tag == "json":
        return extract_decision_json(response_text)
    decision, reason = extract_decision_free(response_text)
    return decision, reason, ""


# ============================================================
# INPUT VALIDATION (before any API call)
# ============================================================

FACTOR_COLUMNS = {  # prompts CSV column -> label meaning "present"/"structured"
    "sensitivity": "Possuir",
    "chain_of_thought": "Possui",
    "prompt_type": "Estruturado",
    "few_shot": "Possuir",
    "output_format": "Estruturado",
}


def load_and_validate_inputs():
    """Reads and checks the prompts CSV and the articles CSV. Prints every
    problem found and stops (SystemExit) if there is any error."""
    errors, warnings = [], []

    # ---- prompts ----
    prompts_df = None
    if not Path(PROMPTS_CSV_PATH).exists():
        errors.append(f"prompts file not found: {PROMPTS_CSV_PATH} (run generate_prompts_complementary.py first)")
    else:
        prompts_df = pd.read_csv(PROMPTS_CSV_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        required = ["run_order", "config_code", "format_tag", "full_prompt"]
        missing_cols = [c for c in required if c not in prompts_df.columns]
        if missing_cols:
            errors.append(f"prompts CSV is missing column(s): {missing_cols}. Columns present: {list(prompts_df.columns)}")
        else:
            codes = prompts_df["config_code"].tolist()
            bad_codes = [c for c in codes if not re.fullmatch(r"[01]{5}", c)]
            if bad_codes:
                errors.append(f"invalid config_code value(s) (expected 5 digits of 0/1): {bad_codes}")
            dup_codes = [c for c, n in Counter(codes).items() if n > 1]
            if dup_codes:
                errors.append(f"duplicated config_code(s): {dup_codes}")
            if set(codes) != set(EXPECTED_CONFIGS) or len(codes) != len(EXPECTED_CONFIGS):
                errors.append(
                    f"config codes found {sorted(codes)} do not match the expected {sorted(EXPECTED_CONFIGS)}"
                )
            try:
                run_orders = [int(x) for x in prompts_df["run_order"]]
                if len(set(run_orders)) != len(run_orders):
                    errors.append(f"duplicated run_order values: {run_orders}")
                prompts_df["run_order"] = run_orders
            except ValueError:
                errors.append(f"run_order must be integers, found: {prompts_df['run_order'].tolist()}")

            for _, row in prompts_df.iterrows():
                code, tag, prompt = row["config_code"], row["format_tag"], row["full_prompt"]
                if tag not in ("json", "free"):
                    errors.append(f"config {code}: format_tag must be 'json' or 'free', found '{tag}'")
                elif re.fullmatch(r"[01]{5}", code) and tag != ("json" if code[4] == "1" else "free"):
                    errors.append(f"config {code}: format_tag '{tag}' contradicts the output-format digit of the code")
                if not prompt.strip():
                    errors.append(f"config {code}: full_prompt is empty")
                elif re.search("[“”‘’]", prompt):
                    warnings.append(
                        f"config {code}: prompt contains curly quotes (typical of an Excel re-save); "
                        f"check that the few-shot/JSON examples were not corrupted"
                    )
                if re.fullmatch(r"[01]{5}", code):
                    for digit, (col, present_label) in zip(code, FACTOR_COLUMNS.items()):
                        if col in prompts_df.columns and (row[col] == present_label) != (digit == "1"):
                            errors.append(
                                f"config {code}: column '{col}' = '{row[col]}' contradicts the code digit '{digit}'"
                            )

    # ---- articles ----
    articles_df = None
    if not Path(ARTICLES_CSV_PATH).exists():
        errors.append(f"articles file not found: {ARTICLES_CSV_PATH}")
    else:
        articles_df = pd.read_csv(ARTICLES_CSV_PATH)
        required = ["key", "title", "abstract", "Gabarito_Humano"]
        missing_cols = [c for c in required if c not in articles_df.columns]
        if missing_cols:
            errors.append(f"articles CSV is missing column(s): {missing_cols}. Columns present: {list(articles_df.columns)}")
        else:
            if len(articles_df) != EXPECTED_N_ARTICLES:
                errors.append(f"articles CSV has {len(articles_df)} rows, expected {EXPECTED_N_ARTICLES}")
            empty_keys = int(articles_df["key"].isna().sum() + (articles_df["key"].astype(str).str.strip() == "").sum())
            if empty_keys:
                errors.append(f"{empty_keys} article(s) with an empty key")
            keys = [str(k) for k in articles_df["key"]]
            dup_keys = [k for k, n in Counter(keys).items() if n > 1]
            if dup_keys:
                errors.append(f"{len(dup_keys)} duplicated article key(s), e.g. {dup_keys[:5]}")
            human = articles_df["Gabarito_Humano"].map(normalize_human_label)
            bad_human = articles_df.loc[human.isna(), "key"].astype(str).tolist()
            if bad_human:
                errors.append(f"{len(bad_human)} article(s) with a missing/unrecognised Gabarito_Humano, e.g. keys {bad_human[:5]}")
            no_title = articles_df.loc[articles_df["title"].isna(), "key"].astype(str).tolist()
            no_abstract = articles_df.loc[articles_df["abstract"].isna(), "key"].astype(str).tolist()
            if no_title:
                warnings.append(f"{len(no_title)} article(s) without title, e.g. {no_title[:5]} (sent as 'nan', as in the original runs)")
            if no_abstract:
                warnings.append(f"{len(no_abstract)} article(s) without abstract, e.g. {no_abstract[:5]} (sent as 'nan', as in the original runs)")

    for w in warnings:
        print(f"[WARNING] {w}")
    if errors:
        print("\n[INPUT VALIDATION FAILED] Nothing was sent to the API:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    human = articles_df["Gabarito_Humano"].map(normalize_human_label)
    print(
        f"Inputs OK: {len(prompts_df)} configurations, {len(articles_df)} articles "
        f"({int((human == 'INCLUDE').sum())} INCLUDE / {int((human == 'EXCLUDE').sum())} EXCLUDE by human consensus)."
    )
    return prompts_df.sort_values("run_order"), articles_df


# ============================================================
# BATCH JSONL BUILDING
# ============================================================


def build_jsonl_text(system_prompt: str, articles_df: pd.DataFrame) -> str:
    lines = []
    for _, article in articles_df.iterrows():
        article_text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
        line = {
            "key": str(article["key"]),
            "request": {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": article_text}]}],
            },
        }
        lines.append(json.dumps(line, ensure_ascii=False) + "\n")
    return "".join(lines)


def build_jsonl(system_prompt: str, articles_df: pd.DataFrame, jsonl_path: Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(build_jsonl_text(system_prompt, articles_df))


# ============================================================
# API RESULT PARSING AND VALIDATION
# ============================================================


def parse_record(record: dict):
    """Returns (response_text, flag). flag is None for a normal, complete
    response, or a FORMAT_ERROR_* tag for API errors / incomplete responses."""
    try:
        candidate = record["response"]["candidates"][0]
        response_text = candidate["content"]["parts"][0]["text"]
        if not isinstance(response_text, str):
            raise TypeError("response text is not a string")
    except (KeyError, IndexError, TypeError, AttributeError):
        return f"[ERROR IN RAW RESPONSE]: {json.dumps(record, ensure_ascii=False)}", "FORMAT_ERROR_API"

    finish_reason = candidate.get("finishReason", candidate.get("finish_reason"))
    if finish_reason is not None and not str(finish_reason).upper().endswith("STOP"):
        return response_text, "FORMAT_ERROR_INCOMPLETE"  # e.g. MAX_TOKENS, SAFETY
    return response_text, None


def parse_api_result(content: str):
    """Parses the JSONL returned by the API. Returns (responses_by_key, stats).
    Duplicated keys are reported (the first one is kept, but the result will
    fail validation)."""
    responses = {}
    stats = {"n_lines": 0, "bad_lines": 0, "records_without_key": 0, "duplicate_keys": []}
    for line in content.splitlines():
        if not line.strip():
            continue
        stats["n_lines"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_lines"] += 1
            continue
        key = record.get("key") if isinstance(record, dict) else None
        if key is None:
            stats["records_without_key"] += 1
            continue
        key = str(key)
        if key in responses:
            stats["duplicate_keys"].append(key)
            continue
        responses[key] = parse_record(record)
    return responses, stats


def build_result_df(articles_df, responses, format_tag, config_code) -> pd.DataFrame:
    rows = []
    for _, article in articles_df.iterrows():
        key = str(article["key"])
        if key in responses:
            response_text, flag = responses[key]
        else:
            response_text, flag = "[NO RESPONSE - KEY NOT FOUND IN RESULT]", "FORMAT_ERROR_MISSING"
        if flag:
            decision, reason, reasoning = flag, response_text, ""
        else:
            decision, reason, reasoning = extract_decision(response_text, format_tag)
        rows.append(
            {
                "model": MODEL,
                "config_code": config_code,
                "key": key,
                "title": article["title"],
                "abstract": article["abstract"],
                "human_consensus": article["Gabarito_Humano"],
                "ai_decision": decision,
                "ai_reason": reason,
                "ai_reasoning": reasoning,
                "raw_response": response_text,
            }
        )
    return pd.DataFrame(rows)


def validate_result_df(df: pd.DataFrame, articles_df: pd.DataFrame, config_code: str, raw_issues=None) -> list:
    """Returns a list of human-readable problems (empty list = valid)."""
    problems = list(raw_issues or [])
    required = ["model", "config_code", "key", "human_consensus", "ai_decision", "raw_response"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return problems + [f"missing column(s): {missing_cols}"]

    expected_keys = [str(k) for k in articles_df["key"]]
    keys = df["key"].astype(str).tolist()

    if len(df) != len(expected_keys):
        problems.append(f"{len(df)} rows, expected {len(expected_keys)}")
    dups = [k for k, n in Counter(keys).items() if n > 1]
    if dups:
        problems.append(f"{len(dups)} duplicated key(s), e.g. {dups[:5]}")
    missing = [k for k in expected_keys if k not in set(keys)]
    if missing:
        problems.append(f"{len(missing)} article(s) without a row, e.g. {missing[:5]}")
    extra = [k for k in set(keys) if k not in set(expected_keys)]
    if extra:
        problems.append(f"{len(extra)} unexpected key(s), e.g. {sorted(extra)[:5]}")

    bad = df.loc[~df["ai_decision"].isin(LABELS)]
    if len(bad):
        counts = Counter(bad["ai_decision"].astype(str))
        problems.append(
            f"{len(bad)} row(s) without a valid decision {dict(counts)}, e.g. keys {bad['key'].astype(str).tolist()[:5]}"
        )

    expected_human = {str(k): as_text(h) for k, h in zip(articles_df["key"], articles_df["Gabarito_Humano"])}
    mismatch = [k for k, h in zip(keys, df["human_consensus"].astype(str)) if k in expected_human and expected_human[k] != h]
    if mismatch:
        problems.append(f"{len(mismatch)} row(s) whose human_consensus differs from the articles file, e.g. {mismatch[:5]}")

    if (df["model"].astype(str) != MODEL).any():
        problems.append(f"'model' column differs from {MODEL}")
    if (df["config_code"].astype(str) != config_code).any():
        problems.append(f"'config_code' column differs from {config_code}")
    return problems


def raw_issues_from_stats(stats: dict, responses: dict, articles_df: pd.DataFrame) -> list:
    issues = []
    if stats["bad_lines"]:
        issues.append(f"{stats['bad_lines']} line(s) of the API result are not valid JSON")
    if stats["records_without_key"]:
        issues.append(f"{stats['records_without_key']} record(s) of the API result without a key")
    if stats["duplicate_keys"]:
        issues.append(f"{len(stats['duplicate_keys'])} duplicated key(s) in the API result, e.g. {stats['duplicate_keys'][:5]}")
    expected = {str(k) for k in articles_df["key"]}
    extra = sorted(k for k in responses if k not in expected)
    if extra:
        issues.append(f"{len(extra)} key(s) in the API result that are not in the articles file, e.g. {extra[:5]}")
    return issues


# ============================================================
# PATHS AND BATCH STATE FILE
# ============================================================


def result_path_for(config_code: str) -> Path:
    return RESULTS_DIR / f"results_config_{config_code}.csv"


def incomplete_path_for(config_code: str) -> Path:
    return RESULTS_DIR / f"incomplete_config_{config_code}.csv"


def state_path_for(config_code: str) -> Path:
    return STATE_DIR / f"batch_state_config_{config_code}.json"


def report_path_for(config_code: str) -> Path:
    return STATE_DIR / f"validation_report_config_{config_code}.json"


def raw_result_path_for(config_code: str) -> Path:
    return RAW_BATCHES_DIR / f"api_result_config_{config_code}.jsonl"


def load_state(config_code: str) -> dict:
    path = state_path_for(config_code)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"  [WARNING] state file {path} is unreadable; it will be archived and a new one started.")
        archive(path)
        return {}


def save_state(config_code: str, state: dict) -> None:
    state["updated_at"] = now_iso()
    atomic_write_json(state_path_for(config_code), state)


def existing_result_status(config_code: str, articles_df: pd.DataFrame):
    """('missing' | 'valid' | 'invalid', problems) for the final results CSV."""
    path = result_path_for(config_code)
    if not path.exists():
        return "missing", []
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as e:  # unreadable / empty file
        return "invalid", [f"could not read {path}: {e}"]
    problems = validate_result_df(df, articles_df, config_code)
    return ("valid" if not problems else "invalid"), problems


# ============================================================
# OBTAIN THE BATCH RESULT (resume / create / poll / download)
# ============================================================


def obtain_batch_result(client, state, config_code, jsonl_path, raw_path) -> str:
    """Makes sure the raw API result is saved at raw_path. Returns 'ok' or 'failed'."""
    batch_job = None

    # 1) Resume a batch that was already recorded (no new batch, no new cost).
    if state.get("batch_name"):
        try:
            batch_job = client.batches.get(name=state["batch_name"])
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 404:
                print(f"  Recorded batch {state['batch_name']} no longer exists on the server; a new one will be created.")
                state["history"].append({"batch_name": state["batch_name"], "state": "NOT_FOUND", "noted_at": now_iso()})
                state["batch_name"] = None
                save_state(config_code, state)
            else:
                raise
        if batch_job is not None:
            current = batch_job.state.name
            if current in FAILED_STATES:
                print(f"  Recorded batch {state['batch_name']} ended in {current}; a new one will be created.")
                state["history"].append({"batch_name": state["batch_name"], "state": current, "noted_at": now_iso()})
                state["batch_name"] = None
                state["batch_state"] = None
                save_state(config_code, state)
                batch_job = None
            else:
                print(f"  Resuming recorded batch {state['batch_name']} (state: {current}). No new batch is created.")

    # 2) Otherwise upload (or reuse the uploaded file) and create the batch.
    if batch_job is None:
        uploaded_name = state.get("uploaded_file")
        if uploaded_name:
            try:
                client.files.get(name=uploaded_name)
                print(f"  Reusing already uploaded file: {uploaded_name}")
            except genai_errors.ClientError:
                uploaded_name = None
        if not uploaded_name:
            uploaded_file = client.files.upload(
                file=str(jsonl_path),
                config={
                    "display_name": f"batch-complementary-{config_code}",
                    "mime_type": "jsonl",
                },
            )
            uploaded_name = uploaded_file.name
            state["uploaded_file"] = uploaded_name
            save_state(config_code, state)
            print(f"  File uploaded: {uploaded_name}")

        while True:
            try:
                batch_job = client.batches.create(
                    model=MODEL,
                    src=uploaded_name,
                    config={"display_name": f"screening-complementary-{config_code}"},
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
        state["batch_name"] = batch_job.name
        state["batch_state"] = batch_job.state.name
        save_state(config_code, state)  # recorded immediately: a restart will resume THIS batch
        print(f"  Job created: {batch_job.name} (initial state: {batch_job.state.name})")

    # 3) Wait for it.
    while batch_job.state.name not in TERMINAL_STATES:
        time.sleep(POLL_INTERVAL_SECONDS)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"  ... current state: {batch_job.state.name}")
        if state.get("batch_state") != batch_job.state.name:
            state["batch_state"] = batch_job.state.name
            save_state(config_code, state)

    if state.get("batch_state") != batch_job.state.name:
        state["batch_state"] = batch_job.state.name
        save_state(config_code, state)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  ERROR: job finished in state {batch_job.state.name}. Skipping to the next one.")
        return "failed"

    # 4) Download and save the JSONL exactly as returned, BEFORE parsing it.
    result_file_name = batch_job.dest.file_name
    content_bytes = client.files.download(file=result_file_name)
    if raw_path.exists():
        archive(raw_path)
    atomic_write_bytes(raw_path, content_bytes)
    state["result_file_name"] = result_file_name
    state["result_saved"] = True
    state["result_saved_at"] = now_iso()
    save_state(config_code, state)
    print(f"  Raw API result saved: {raw_path} ({len(content_bytes)} bytes)")
    return "ok"


# ============================================================
# RUN ONE FULL CONFIGURATION
# ============================================================


def run_config(client, prompt_row: pd.Series, articles_df: pd.DataFrame, total_configs: int, model_info: dict) -> str:
    """Returns 'complete', 'failed' (batch did not succeed) or 'invalid'
    (results did not pass validation)."""
    config_code = str(prompt_row["config_code"])
    run_order = int(prompt_row["run_order"])
    system_prompt = prompt_row["full_prompt"]
    format_tag = prompt_row["format_tag"]

    # Resume support: the final CSV only counts if it passes validation.
    status, problems = existing_result_status(config_code, articles_df)
    if status == "valid":
        print(
            f"\nRun {run_order}/{total_configs} -- Config {config_code}: "
            f"result already exists and passed validation ({result_path_for(config_code)}). Skipping."
        )
        return "complete"
    if status == "invalid":
        moved = archive(result_path_for(config_code))
        print(f"\n  [WARNING] The existing result for config {config_code} FAILED validation and is not accepted:")
        for p in problems[:10]:
            print(f"     - {p}")
        print(f"     (moved to {moved})")

    print(
        f"\n{'='*60}\n"
        f"Run {run_order}/{total_configs} -- Config {config_code} "
        f"(format: {format_tag}) -- model {MODEL}\n{'='*60}"
    )

    jsonl_path = RAW_BATCHES_DIR / f"batch_config_{config_code}.jsonl"
    build_jsonl(system_prompt, articles_df, jsonl_path)
    request_sha = sha256_file(jsonl_path)
    print(f"  JSONL built: {jsonl_path} ({len(articles_df)} articles)")

    # Batch state: only reused if it belongs to this exact model + prompt + articles.
    state = load_state(config_code)
    if state and (state.get("model") != MODEL or state.get("request_sha256") != request_sha):
        moved = archive(state_path_for(config_code))
        print(f"  [WARNING] The recorded batch state was for a different model/prompt/articles; archived as {moved}. Starting fresh.")
        state = {}
    if not state:
        state = {
            "model": MODEL,
            "config_code": config_code,
            "model_info": model_info,
            "prompt_sha256": sha256_text(system_prompt),
            "request_sha256": request_sha,
            "n_requests": len(articles_df),
            "created_at": now_iso(),
            "uploaded_file": None,
            "batch_name": None,
            "batch_state": None,
            "result_file_name": None,
            "result_saved": False,
            "status": "in_progress",
            "history": [],
        }
        save_state(config_code, state)

    raw_path = raw_result_path_for(config_code)
    if state.get("result_saved") and raw_path.exists():
        print("  Reusing the saved API result (no API calls).")
    else:
        state["result_saved"] = False
        outcome = obtain_batch_result(client, state, config_code, jsonl_path, raw_path)
        if outcome != "ok":
            state["status"] = "failed"
            save_state(config_code, state)
            return "failed"

    # Parse the saved JSONL and validate the 290 decisions.
    content = raw_path.read_bytes().decode("utf-8")
    responses, stats = parse_api_result(content)
    result_df = build_result_df(articles_df, responses, format_tag, config_code)
    problems = validate_result_df(
        result_df, articles_df, config_code, raw_issues_from_stats(stats, responses, articles_df)
    )

    n_errors = result_df["ai_decision"].str.startswith("FORMAT_ERROR").sum()
    print(f"  Formatting errors: {n_errors} out of {len(result_df)} articles")

    if problems:
        atomic_write_csv(result_df, incomplete_path_for(config_code))
        atomic_write_json(
            report_path_for(config_code),
            {
                "config_code": config_code,
                "model": MODEL,
                "checked_at": now_iso(),
                "ok": False,
                "problems": problems,
                "decision_counts": dict(Counter(result_df["ai_decision"])),
                "api_result_stats": stats,
            },
        )
        state["status"] = "invalid"
        save_state(config_code, state)
        print(f"  [INVALID] Config {config_code} is NOT complete. Problems found:")
        for p in problems:
            print(f"     - {p}")
        print(f"  All rows saved for inspection: {incomplete_path_for(config_code)}")
        print(f"  Report: {report_path_for(config_code)}")
        print(f"  The raw API result was kept ({raw_path}). Running again re-checks it WITHOUT new API calls.")
        print(f"  To send this configuration again, move/delete {state_path_for(config_code)} (older files are archived, never overwritten).")
        return "invalid"

    atomic_write_csv(result_df, result_path_for(config_code))
    for stale in (incomplete_path_for(config_code), report_path_for(config_code)):
        if stale.exists():
            stale.unlink()
    state["status"] = "complete"
    save_state(config_code, state)
    print(f"  Done. All {len(result_df)} decisions validated. Results saved to: {result_path_for(config_code)}")
    return "complete"


# ============================================================
# MODEL CHECK AND RUN PLAN
# ============================================================


def check_model(client) -> dict:
    """Confirms the model ID exists (models.get is free) and returns its info."""
    try:
        info = client.models.get(model=MODEL)
    except genai_errors.ClientError as e:
        if getattr(e, "code", None) == 404:
            raise SystemExit(f"[ERROR] Model '{MODEL}' was not found by the API. Check MODEL at the top of this script.")
        print(f"  [WARNING] Could not verify the model ID ({e}); continuing.")
        return {"id": MODEL, "verified": False}
    except Exception as e:
        print(f"  [WARNING] Could not verify the model ID ({e}); continuing.")
        return {"id": MODEL, "verified": False}

    model_info = {
        "id": MODEL,
        "verified": True,
        "name": getattr(info, "name", None),
        "display_name": getattr(info, "display_name", None),
        "version": getattr(info, "version", None),
    }
    actions = getattr(info, "supported_actions", None)
    print(f"Model check: {model_info['name']} | {model_info['display_name']} | version {model_info['version']}")
    if actions and "batchGenerateContent" not in actions:
        print(f"  [WARNING] The API does not list 'batchGenerateContent' for this model (supported: {actions}).")
    return model_info


def describe_plan(config_code, prompt_row, articles_df):
    """What a run would do for this configuration, and how many new calls."""
    sha = sha256_text(build_jsonl_text(prompt_row["full_prompt"], articles_df))
    st = load_state(config_code)
    if st and st.get("model") == MODEL and st.get("request_sha256") == sha:
        if st.get("result_saved") and raw_result_path_for(config_code).exists():
            return "re-check the saved API result (no API calls)", 0
        if st.get("batch_name"):
            return f"resume recorded batch {st['batch_name']} (a new batch only if it failed/expired)", 0
    return "send a NEW batch", len(articles_df)


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

    prompts_df, articles_df = load_and_validate_inputs()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}")
    print("Run order (config_code):", prompts_df["config_code"].tolist())

    client = genai.Client(api_key=api_key)
    model_info = check_model(client)

    pending = []
    for _, row in prompts_df.iterrows():
        status, _ = existing_result_status(row["config_code"], articles_df)
        if status != "valid":
            pending.append(row)
    if not pending:
        print("\nAll configurations already have validated results. Nothing to do.")
        return

    print("\nPlan:")
    new_calls = 0
    for row in pending:
        action, n_calls = describe_plan(row["config_code"], row, articles_df)
        new_calls += n_calls
        print(f"  - config {row['config_code']}: {action}")
    print(f"New classifications to be requested: {new_calls} (1 vote per article, model '{MODEL}').")

    if REQUIRE_CONFIRMATION:
        if input("Type YES to continue: ").strip().upper() != "YES":
            raise SystemExit("Cancelled. Nothing was sent to the API.")

    statuses = {}
    for _, prompt_row in prompts_df.iterrows():
        statuses[prompt_row["config_code"]] = run_config(client, prompt_row, articles_df, len(prompts_df), model_info)

    print("\nSummary:")
    for code, status in statuses.items():
        print(f"  config {code}: {status}")
    if all(s == "complete" for s in statuses.values()):
        print("\nAll configurations are complete and validated.")
    else:
        print("\nSome configurations are NOT complete (see above). Nothing was marked as successful without validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
