"""Load the Olist CSVs into a DuckDB database file.

Run once from the repo root:
    python load_data.py """

from pathlib import Path

import duckdb

HERE = Path(__file__).parent
CSV_DIR = HERE / "archive"
DB_PATH = HERE / "olist.duckdb"


TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}


def load():
    con = duckdb.connect(str(DB_PATH))

    for table, csv_file in TABLES.items():
        csv_path = CSV_DIR / csv_file
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}')"
        )

        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:30s} {rows:>8,} rows")

    con.close()
    print(f"\nWrote {DB_PATH}")


if __name__ == "__main__":
    load()
