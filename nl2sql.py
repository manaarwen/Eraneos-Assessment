"""The natural-language -> SQL -> answer loop."""

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from schema import SCHEMA_DESCRIPTION
from validate import validate_sql

load_dotenv()

MODEL = "claude-sonnet-5"
DB_PATH = Path(__file__).parent / "olist.duckdb"
MAX_ATTEMPTS = 3
MAX_TOKENS = 4096

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class TruncatedResponse(Exception):
    """The model hit max_tokens before finishing."""


def generate_sql(question: str, repair: tuple[str, str] | None = None) -> str:
    """Ask the model for a SQL query that answers `question`."""
    if repair is None:
        user_message = question
    else:
        failed_sql, error = repair
        user_message = (
            f"Question: {question}\n\n"
            f"This SQL was tried and did not work:\n{failed_sql}\n\n"
            f"Error:\n{error}\n\n"
            "Fix the SQL so it correctly answers the question."
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SCHEMA_DESCRIPTION,
        messages=[{"role": "user", "content": user_message}],
    )

    if response.stop_reason == "max_tokens":
        raise TruncatedResponse(
            f"Model response was cut off at max_tokens ({MAX_TOKENS})."
        )

    text = "".join(block.text for block in response.content if block.type == "text")
    return _strip_code_fence(text.strip())


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing ``` or ```sql fence, if present."""
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop the opening ``` or ```sql line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def run_sql(sql: str) -> pd.DataFrame:
    """Run `sql` against the DuckDB file and return the result as a DataFrame."""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    """Ask the model to explain the query result in plain English. """
    if len(df) <= 20:
        preview = df.head(20).to_string(index=False)
        scope_note = ""
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        parts = [f"First 10 rows (in the query's own order):\n{df.head(10).to_string(index=False)}"]
        if len(numeric_cols) > 0:
            metric = numeric_cols[-1]
            parts.append(f"5 highest by {metric}:\n{df.nlargest(5, metric).to_string(index=False)}")
            parts.append(f"5 lowest by {metric}:\n{df.nsmallest(5, metric).to_string(index=False)}")
        preview = "\n\n".join(parts)
        scope_note = (
            f"\nIMPORTANT: this result has {len(df)} rows in total. You have only been "
            "shown the 10 above plus the 5 highest and 5 lowest, not the full result. Do "
            "not state which row is the overall highest, lowest, first, or last unless it "
            "appears in what you were shown. Say the answer isn't visible in what you "
            "were given instead of guessing."
        )

    prompt = (
        f"Question: {question}\n\n"
        f"SQL that was run:\n{sql}\n\n"
        f"{scope_note}\n"
        f"Result:\n{preview}\n\n"
        "In plain English, answer the original question using this result. "
        "Be concise and specific about the numbers."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text").strip()

    if response.stop_reason == "max_tokens":
        text += f"\n\n_(Explanation cut off at max_tokens ({MAX_TOKENS}).)_"

    return text


@dataclass
class Answer:
    """Everything the UI needs to render one question's outcome. """

    sql: str
    df: pd.DataFrame | None
    attempts: list[dict]
    error: str | None


def _log_attempt(n: int, sql: str, outcome: str, detail: str | None) -> None:
    """Print one attempt to stdout so every attempt is visible while the
    app runs, not just the one that succeeded.
    """
    print(f"--- attempt {n}: {outcome} ---")
    print(sql)
    if detail:
        print(f"detail: {detail}")


def answer_question(question: str) -> Answer:
    """Generate SQL, validate it, run it. Retry with the failure fed
    back to the model as repair context, up to MAX_ATTEMPTS times total."""

    print(f"=== question: {question} ===")

    attempts = []
    repair = None
    sql = None

    for n in range(1, MAX_ATTEMPTS + 1):
        try:
            sql = generate_sql(question, repair=repair)
        except TruncatedResponse as e:
            detail = str(e)
            _log_attempt(n, "(no SQL returned)", "truncated", detail)
            attempts.append(
                {"n": n, "sql": "(no SQL returned)", "outcome": "truncated", "detail": detail}
            )
            repair = None
            continue

        errors = validate_sql(sql)
        if errors:
            detail = "; ".join(errors)
            _log_attempt(n, sql, "rejected", detail)
            attempts.append({"n": n, "sql": sql, "outcome": "rejected", "detail": detail})
            repair = (sql, detail)
            continue

        try:
            df = run_sql(sql)
        except Exception as e:
            detail = str(e)
            _log_attempt(n, sql, "failed", detail)
            attempts.append({"n": n, "sql": sql, "outcome": "failed", "detail": detail})
            repair = (sql, detail)
            continue

        _log_attempt(n, sql, "ok", None)
        attempts.append({"n": n, "sql": sql, "outcome": "ok", "detail": None})
        return Answer(sql=sql, df=df, attempts=attempts, error=None)

    last_detail = attempts[-1]["detail"] if attempts else None
    error = f"Could not produce a working query after {MAX_ATTEMPTS} attempts. Last error: {last_detail}"
    return Answer(sql=sql, df=None, attempts=attempts, error=error)
