"""Gate that validates SQL before it is sent to DuckDB. (Prevents the model from executing)"""

from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

DB_PATH = Path(__file__).parent / "olist.duckdb"


_schema_cache = None


def _load_schema() -> tuple[set[str], set[str]]:
    """Return (table_names, column_names), both lowercased, introspected
    from the live database so this can never drift from the real schema."""
    global _schema_cache
    if _schema_cache is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            rows = con.execute(
                "SELECT table_name, column_name FROM duckdb_columns() "
                "WHERE NOT internal"
            ).fetchall()
        finally:
            con.close()
        tables = {t.lower() for t, _ in rows}
        columns = {c.lower() for _, c in rows}
        _schema_cache = (tables, columns)
    return _schema_cache


def validate_sql(sql: str) -> list[str]:
    """Return reasons `sql` must not be executed. Empty list means OK."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.SqlglotError as e:
        return [f"SQL does not parse: {e}"]

    if len(statements) != 1:
        return [f"Expected exactly one statement, got {len(statements)}"]

    stmt = statements[0]
    if stmt is None:
        return ["Model returned no SQL at all (empty response)"]

    if not isinstance(stmt, (exp.Select, exp.SetOperation, exp.Subquery)):
        return [f"Only SELECT/WITH queries are allowed, got {type(stmt).__name__}"]

    tables, columns = _load_schema()

    
    cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE) if cte.alias}
    alias_names = {a.alias.lower() for a in stmt.find_all(exp.Alias) if a.alias}

    reasons = []
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if not name:
            reasons.append(
                "Table functions are not allowed - query the loaded tables by name"
            )
            continue
        if name not in tables and name not in cte_names:
            reasons.append(f"Unknown table: {name}")

    for column in stmt.find_all(exp.Column):
        name = column.name.lower()
        if not name or name == "*":
            continue
        if name not in columns and name not in alias_names:
            reasons.append(f"Unknown column: {name}")

    return reasons
