"""The natural-language -> SQL -> answer loop.

Phase 2a adds a validation gate (validate.py) and a retry/repair loop on top
of the Phase 1 functions below, which are otherwise unchanged: generate_sql
still just asks the model for SQL, run_sql still just executes it with no
try/except of its own. answer_question is the new orchestrator - it's the
one function that knows about retries; generate_sql, run_sql, and
explain_result stay simple and single-purpose so a failure is still easy to
place: did generate_sql return garbage, did run_sql blow up, or did
explain_result get confused.
"""

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
# Anchored to this file's directory, matching load_data.py, so run_sql finds
# the database regardless of the process's working directory - a bare
# "olist.duckdb" would fail with a confusing "unable to open database file"
# if streamlit (or any launcher) starts from somewhere other than the repo
# root.
DB_PATH = Path(__file__).parent / "olist.duckdb"

# Initial attempt + 2 repairs. Three tries in a row failing is treated as
# the question being genuinely unanswerable for now (bad phrasing, no
# matching data) rather than a transient slip worth retrying further -
# see DESIGN_DECISIONS.md for why 2 retries and not more.
MAX_ATTEMPTS = 3

# 1024 was not enough. claude-sonnet-5 thinks before answering by default, and
# that thinking is billed against max_tokens - so on a multi-part question
# ("average review score, order count, delivery time and late share per
# category") the whole budget went on a thinking block and the reply came back
# with stop_reason="max_tokens" and the SQL cut in half, or missing entirely.
# That is the real source of the "model returned an empty response" case
# validate.py documents. The SQL itself is never long; the headroom is for the
# reasoning in front of it.
MAX_TOKENS = 4096

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class TruncatedResponse(Exception):
    """The model hit max_tokens before finishing.

    Its own exception type because it needs different handling from a bad
    query: the output isn't wrong, it's incomplete, and retrying with a
    *longer* repair prompt makes it strictly more likely to happen again.
    Worth naming so the attempt log says "truncated" rather than reporting a
    confusing parse error on half a query.
    """


def generate_sql(question: str, repair: tuple[str, str] | None = None) -> str:
    """Ask the model for a SQL query that answers `question`.

    The model is asked to return SQL only, no prose. In practice it
    sometimes wraps the query in a ```sql ... ``` fence anyway, so we strip
    that off. We do NOT try to strip stray prose before/after the query -
    if the model adds commentary, run_sql will fail on it and that failure
    is exactly the kind of thing Phase 1 is meant to surface, not hide.

    `repair`, when given, is a (failed_sql, error) pair from a previous
    attempt. When present, the user message includes that SQL and the error
    text and asks the model to fix it, instead of sending just the
    question. The system prompt (the schema description) doesn't change -
    what's wrong isn't the schema, it's the specific query, so only the
    user message needs the extra context.
    """
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

    # Checked explicitly rather than trusting the text: a truncated response
    # still contains *something*, and half a query is exactly the kind of
    # failure that looks like a model mistake when it is really our token
    # budget. Without this the loop retries a truncation with a longer prompt.
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
    """Run `sql` against the DuckDB file and return the result as a DataFrame.

    Opened read-only so a generated DROP or UPDATE fails loudly instead of
    damaging the database - that's a safety rail, not a validation layer.
    No try/except: a bad query should raise and be handled by the caller
    (the UI), not swallowed here.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    """Ask the model to explain the query result in plain English.

    This is a second, separate API call because the model can't write a
    grounded explanation of numbers it hasn't seen yet - asking for SQL and
    an explanation in one response means the explanation gets written before
    the query has actually run.

    Sends the whole result when it's 20 rows or fewer. Above that, sends the
    first 10 (in the query's own order) plus the 5 highest and 5 lowest by
    the result's numeric column, instead of a plain head(20) - see
    DESIGN_DECISIONS.md ("Explanation row cutoff: head + extremes, not just
    head(20)") for the fabricated-superlative case that motivated this. This
    is also the one place in the app where real data (not just schema)
    leaves the machine - see DESIGN_DECISIONS.md ("What leaves the machine")
    for why that's fine for this public dataset and what would need to
    change for real company data.
    """
    if len(df) <= 20:
        preview = df.head(20).to_string(index=False)
        scope_note = ""
    else:
        # The last numeric column is a heuristic for "the metric this
        # question is actually about" - true for every result shape this
        # app produces (a count, an average, a total, computed last after
        # whatever it's grouped by). A result with two meaningful numeric
        # columns only gets extremes on this one; see DESIGN_DECISIONS.md
        # for that residual gap.
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
            "appears in what you were shown - say the answer isn't visible in what you "
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

    # Same truncation risk as generate_sql, but this one is not inside the
    # retry loop, so it can't be repaired - all it can do is not pretend the
    # answer is complete. A silently half-written explanation is worse than a
    # complete one with a visible warning on the end.
    if response.stop_reason == "max_tokens":
        text += f"\n\n_(Explanation cut off at max_tokens ({MAX_TOKENS}).)_"

    return text


@dataclass
class Answer:
    """Everything the UI needs to render one question's outcome.

    A plain data container, not an abstraction layer - four related values
    returned together because answer_question has four things to report:
    the SQL it ended up with, the result (None if every attempt failed),
    the full attempt log for the UI's expander, and an error message set
    only on failure.
    """

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
    """Generate SQL, validate it, run it - retrying with the failure fed
    back to the model as repair context, up to MAX_ATTEMPTS times total.

    A query can fail two different ways, and both are treated the same way
    by this loop even though they come from different places: validate_sql
    catches SQL that doesn't parse, isn't a SELECT, or references an
    unknown name, before it ever reaches the database; run_sql can still
    raise on SQL that passed validation but is wrong in a way validate.py
    doesn't check (e.g. a column that exists but on the wrong table - see
    the "global column check" limitation in validate.py). Either way, the
    SQL and the error text become the `repair` argument for the next
    generate_sql call, so the model sees exactly what went wrong.

    Returns an Answer with `error` set rather than raising when all
    attempts fail - an unanswerable question is a normal outcome the UI
    should render honestly, not an exception to handle.
    """
    # Printed once here rather than passed into _log_attempt, so the stdout
    # blocks below are tied to the question that produced them. One line
    # instead of four call-site changes, and safe because the app is a single
    # process - two turns can't interleave in the terminal.
    print(f"=== question: {question} ===")

    attempts = []
    repair = None
    sql = None

    for n in range(1, MAX_ATTEMPTS + 1):
        try:
            sql = generate_sql(question, repair=repair)
        except TruncatedResponse as e:
            # Retrying a truncation with the repair context attached would make
            # the prompt longer and the truncation more likely, so this drops
            # the repair context and tries the plain question again instead.
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
