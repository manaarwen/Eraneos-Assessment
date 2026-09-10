"""Predict order review scores in two modes: at-purchase and post-delivery."""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nl2sql import MAX_TOKENS, MODEL, TruncatedResponse, client

DB_PATH = Path(__file__).parent / "olist.duckdb"
TRAIN_FRACTION = 0.8


FEATURE_SQL = """
WITH review AS (
    SELECT order_id, MIN(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
items AS (
    SELECT order_id,
           SUM(price) AS price_total,
           COUNT(*)   AS n_items
    FROM order_items
    GROUP BY order_id
),
payments AS (
    SELECT order_id, MAX(payment_installments) AS installments
    FROM order_payments
    GROUP BY order_id
),
primary_seller AS (
    SELECT order_id, seller_id FROM (
        SELECT oi.order_id, oi.seller_id,
               ROW_NUMBER() OVER (PARTITION BY oi.order_id ORDER BY oi.price DESC) AS rn
        FROM order_items oi
    ) WHERE rn = 1
)
SELECT
    o.order_purchase_timestamp,
    CAST(r.review_score <= 3 AS INTEGER) AS bad_review,
    r.review_score,
    DATE_DIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date)     AS promised_days,
    DATE_DIFF('day', o.order_purchase_timestamp, o.order_delivered_customer_date)     AS delivery_days,
    DATE_DIFF('day', o.order_estimated_delivery_date, o.order_delivered_customer_date) AS days_late,
    i.price_total,
    i.n_items,
    p.installments,
    ps.seller_id
FROM orders o
JOIN review         r  ON r.order_id  = o.order_id
JOIN items          i  ON i.order_id  = o.order_id
JOIN payments       p  ON p.order_id  = o.order_id
JOIN primary_seller ps ON ps.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
"""

AT_PURCHASE_FEATURES = [
    "promised_days",
    "log_price",
    "n_items",
    "installments",
    "seller_hist_bad",
]
POST_DELIVERY_FEATURES = AT_PURCHASE_FEATURES + [
    "delivery_days",
    "is_late",
]


AT_PURCHASE_USER_FEATURES = [
    "promised_days",
    "price_total",
    "n_items",
    "installments",
    "seller_hist_bad",
]
POST_DELIVERY_USER_FEATURES = AT_PURCHASE_USER_FEATURES + ["delivery_days", "days_late"]
USER_FEATURES = POST_DELIVERY_USER_FEATURES

# Mentioned often enough in questions to be worth recognising, but not model
# inputs: each one was measured on the holdout and did not earn its place.
IGNORED_FEATURES = ["product_category", "customer_state", "freight_total", "same_state"]


FEATURE_DESCRIPTIONS = {
    "promised_days": "Days from purchase to the promised delivery date",
    "price_total": "Total order value in reais (R$), all items",
    "n_items": "Number of items on the order",
    "installments": "Number of payment installments",
    "seller_hist_bad": "Seller's past bad-review rate, 0 to 1 (0.2 is typical)",
    "delivery_days": "Days the delivery actually took, purchase to arrival",
    "days_late": "Days later than promised - negative means it arrived early",
}

_frame_cache = None
_model_cache = {}


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the model is actually fit on."""
    df = df.copy()
    df["log_price"] = np.log1p(df["price_total"])
    df["is_late"] = (df["days_late"] > 0).astype(int)
    return df


def _add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add seller_hist_bad: the share of this seller's EARLIER orders that got
    a bad review."""
    df = df.copy()
    prior = df.groupby("seller_id")["bad_review"]
    df["seller_hist_bad"] = prior.transform(lambda s: s.shift(1).expanding().mean())
    train_rate = df["bad_review"].iloc[: int(len(df) * TRAIN_FRACTION)].mean()
    df["seller_hist_bad"] = df["seller_hist_bad"].fillna(train_rate)
    return df


def load_frame() -> pd.DataFrame:
    """Run FEATURE_SQL and return the derived training frame, sorted oldest
    first.
    """
    global _frame_cache
    if _frame_cache is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute(FEATURE_SQL).df()
        finally:
            con.close()
        df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
        _frame_cache = _add_derived(_add_seller_history(df))
    return _frame_cache


def get_model(mode: str):
    """Return (fitted_pipeline, medians) for `mode`, training on first call. """
    if mode not in _model_cache:
        features = _features_for(mode)
        df = load_frame()
        split = int(len(df) * TRAIN_FRACTION)
        train = df.iloc[:split]

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pipeline.fit(train[features], train["bad_review"])

        medians = train.median(numeric_only=True)
        _model_cache[mode] = (pipeline, medians)
    return _model_cache[mode]


def _features_for(mode: str) -> list[str]:
    return POST_DELIVERY_FEATURES if mode == "post_delivery" else AT_PURCHASE_FEATURES


_auc_cache = {}
_lift_cache = {}

#Percentage of orders assumed the business can afford to act on
BUDGET = 0.10


def _holdout_scores(mode: str):
    """Predicted risk for every holdout order, plus the holdout frame itself."""
    pipeline, _ = get_model(mode)
    df = load_frame()
    test = df.iloc[int(len(df) * TRAIN_FRACTION) :]
    return pipeline.predict_proba(test[_features_for(mode)])[:, 1], test


def holdout_auc(mode: str) -> float:
    """Holdout ROC-AUC for `mode`, cached after first call. Used in the UI to report model quality."""
    if mode not in _auc_cache:
        from sklearn.metrics import roc_auc_score

        p, test = _holdout_scores(mode)
        _auc_cache[mode] = roc_auc_score(test["bad_review"], p)
    return _auc_cache[mode]


def holdout_lift(mode: str, budget: float = BUDGET) -> dict:
    """What a `budget`-sized intervention list would actually catch.

    AUC asks a question nobody acts on ("rank a random good order against a
    random bad one"). This ranks the holdout by predicted risk, takes the
    riskiest `budget` share, and reports what is inside that slice - which is
    the decision a business actually makes when it can only afford to call,
    expedite or comp a limited number of orders.
    """
    key = (mode, budget)
    if key not in _lift_cache:
        p, test = _holdout_scores(mode)
        n = int(len(test) * budget)
        flagged = np.argsort(-p)[:n]

        bad = test["bad_review"].values
        one_star = (test["review_score"].values == 1).astype(int)
        base_rate = bad.mean()
        precision = bad[flagged].mean()

        _lift_cache[key] = {
            "budget": budget,
            "n_flagged": n,
            "n_total": len(test),
            "base_rate": base_rate,
            "precision": precision,
            "recall": bad[flagged].sum() / bad.sum(),
            "lift": precision / base_rate,
            "one_star_recall": one_star[flagged].sum() / one_star.sum(),
        }
    return _lift_cache[key]


@dataclass
class Prediction:
    """Everything the UI needs to render one prediction honestly."""

    mode: str
    probability: float
    inputs: dict = field(default_factory=dict)
    provided: list[str] = field(default_factory=list)
    defaulted: list[str] = field(default_factory=list)
    ignored: dict = field(default_factory=dict)
    out_of_range: list[str] = field(default_factory=list)
    error: str | None = None


def _training_bounds() -> dict:
    """1st/99th percentile of each user-facing feature over the training slice.
    Used to flag values that are extreme relative to the training data, so the UI can warn the user that the model is extrapolating."""
    df = load_frame()
    train = df.iloc[: int(len(df) * TRAIN_FRACTION)]
    return {f: (train[f].quantile(0.01), train[f].quantile(0.99)) for f in USER_FEATURES}


def _resolve_inputs(stated: dict, medians) -> dict:
    """Fill in a complete feature row from whatever the question stated."""
    row = {f: float(medians[f]) for f in USER_FEATURES}

    for f in USER_FEATURES:
        if stated.get(f) is not None:
            row[f] = float(stated[f])

    if stated.get("days_late") is not None:
        row["delivery_days"] = row["promised_days"] + row["days_late"]
    elif stated.get("delivery_days") is not None:
        row["days_late"] = row["delivery_days"] - row["promised_days"]

    return row


def predict(stated: dict, ignored: dict | None = None) -> Prediction:
    """Score one hypothetical order."""
    mode = (
        "post_delivery"
        if stated.get("days_late") is not None or stated.get("delivery_days") is not None
        else "at_purchase"
    )
    pipeline, medians = get_model(mode)

    row = _resolve_inputs(stated, medians)
    features = _features_for(mode)
    X = _add_derived(pd.DataFrame([row]))[features]
    probability = float(pipeline.predict_proba(X)[0, 1])

    # Report against this mode's inputs only, so nothing appears in the inputs table that the model didn't actually use.
    user_features = (
        POST_DELIVERY_USER_FEATURES if mode == "post_delivery" else AT_PURCHASE_USER_FEATURES
    )
    #Track which features were derived from others, used in UI
    derived = set()
    if stated.get("days_late") is not None:
        derived.add("delivery_days")
    elif stated.get("delivery_days") is not None:
        derived.add("days_late")

    provided = sorted(
        f for f in user_features if stated.get(f) is not None and f not in derived
    )
    defaulted = sorted(set(user_features) - set(provided) - derived)
    bounds = _training_bounds()
    out_of_range = [f for f in provided if not bounds[f][0] <= row[f] <= bounds[f][1]]

    return Prediction(
        mode=mode,
        probability=probability,
        inputs={f: row[f] for f in user_features},
        provided=provided,
        defaulted=defaulted,
        ignored=ignored or {},
        out_of_range=out_of_range,
    )


# ---------------------------------------------------------------------------
# Turning a typed question into model inputs
# ---------------------------------------------------------------------------

EXTRACT_TOOL = {
    "name": "review_score_scenario",
    "description": (
        "Record the order details stated in the user's question, for predicting "
        "the review score. Only fill in fields the question actually states. "
        "Leave everything else out entirely - do not guess typical values."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "promised_days": {
                "type": "number",
                "description": "Days between purchase and the PROMISED/estimated delivery date.",
            },
            "delivery_days": {
                "type": "number",
                "description": "Days the delivery ACTUALLY took, from purchase to arrival.",
            },
            "days_late": {
                "type": "number",
                "description": (
                    "Days later than promised. Positive means late, negative means early. "
                    "'arrived 5 days early' is -5."
                ),
            },
            "price_total": {"type": "number", "description": "Order value in Brazilian reais."},
            "freight_total": {"type": "number", "description": "Shipping cost in reais."},
            "n_items": {"type": "number", "description": "Number of items on the order."},
            "installments": {"type": "number", "description": "Number of payment installments."},
            "same_state": {
                "type": "number",
                "description": "1 if seller and customer are in the same state, 0 if not.",
            },
            "seller_hist_bad": {
                "type": "number",
                "description": (
                    "The seller's track record: the share of their past orders that got a "
                    "bad review, as a fraction between 0 and 1. 'a seller with a 30% bad "
                    "review rate' is 0.3; 'a highly rated seller' is about 0.05; 'a poorly "
                    "rated seller' is about 0.4. The dataset average is about 0.2."
                ),
            },
            "product_category": {"type": "string", "description": "Product category, if mentioned."},
            "customer_state": {"type": "string", "description": "Brazilian state code, if mentioned."},
        },
        "required": [],
    },
}

EXTRACT_SYSTEM = (
    "You extract order details from questions about predicting Olist review scores. "
    "Call the review_score_scenario tool with ONLY the values the question actually "
    "states. Omit any field the question does not mention - it will be filled with a "
    "dataset median. Never guess."
)


def extract_features(question: str) -> tuple[dict, dict]:
    """Pull stated order details out of `question` using a tool call.
    Returns (stated, ignored): values the model can use, and values the
    question mentioned that are not model inputs (category, state), kept
    separately so the UI can say they were dropped."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=EXTRACT_SYSTEM,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": EXTRACT_TOOL["name"]},
        messages=[{"role": "user", "content": question}],
    )


    if response.stop_reason == "max_tokens":
        raise TruncatedResponse(
            f"Feature extraction was cut off at max_tokens ({MAX_TOKENS})."
        )

    payload = {}
    for block in response.content:
        if block.type == "tool_use":
            payload = block.input
            break
        
    stated = {k: v for k, v in payload.items() if k in USER_FEATURES and v is not None}
    ignored = {
        k: v
        for k, v in payload.items()
        if k in IGNORED_FEATURES and v is not None and not _is_placeholder(v)
    }
    return stated, ignored

#for ignoring placeholders
_PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "null", "not specified", "not mentioned"}


def _is_placeholder(value) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().strip("<>").lower() in _PLACEHOLDERS


def answer_prediction(question: str) -> Prediction:
    """Question in, Prediction out."""
    stated, ignored = extract_features(question)
    print(f"--- extracted: stated={stated} ignored={ignored} ---")

    if not stated:
        return Prediction(
            mode="none",
            probability=0.0,
            ignored=ignored,
            error=(
                "No order details could be read from that question. Try naming "
                "something concrete, e.g. 'a R$150 order that arrives 10 days late'."
            ),
        )

    return predict(stated, ignored)


# ---------------------------------------------------------------------------
# Evaluation report: python predict.py
# ---------------------------------------------------------------------------


def _report() -> None:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    df = load_frame()
    split = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:split], df.iloc[split:]

    print(f"rows {len(df):,}   train {len(train):,}   holdout {len(test):,}")
    print(f"train period {train.order_purchase_timestamp.min():%Y-%m-%d} -> "
          f"{train.order_purchase_timestamp.max():%Y-%m-%d}")
    print(f"holdout period {test.order_purchase_timestamp.min():%Y-%m-%d} -> "
          f"{test.order_purchase_timestamp.max():%Y-%m-%d}")
    print(f"bad-review rate: train {train.bad_review.mean():.3f}   holdout {test.bad_review.mean():.3f}")
    print()

    for mode in ("at_purchase", "post_delivery"):
        pipeline, _ = get_model(mode)
        features = _features_for(mode)
        p = pipeline.predict_proba(test[features])[:, 1]
        y = test["bad_review"]

        print(f"=== {mode} ===")
        print(f"  ROC-AUC   {roc_auc_score(y, p):.4f}   (0.5 = coin flip)")
        print(f"  PR-AUC    {average_precision_score(y, p):.4f}   (baseline = {y.mean():.4f})")
        print(f"  Brier     {brier_score_loss(y, p):.4f}   "
              f"(predicting the base rate every time = {brier_score_loss(y, np.full(len(y), train.bad_review.mean())):.4f})")

        print("  coefficients (standardised, + = more likely bad):")
        lr = pipeline.named_steps["logisticregression"]
        for name, coef in sorted(zip(features, lr.coef_[0]), key=lambda t: -abs(t[1])):
            print(f"    {name:16s} {coef:+.3f}")

        lift = holdout_lift(mode)
        print(f"  at a {lift['budget']:.0%} intervention budget "
              f"({lift['n_flagged']:,} of {lift['n_total']:,} orders):")
        print(f"    precision {lift['precision']:.3f}   (random targeting = {lift['base_rate']:.3f})")
        print(f"    lift      {lift['lift']:.2f}x  - times better than acting at random")
        print(f"    recall    {lift['recall']:.3f}   of all bad reviews land in this slice")
        print(f"    1-star    {lift['one_star_recall']:.3f}   of all 1-star reviews land in this slice")

        print("  calibration (predicted vs observed, by decile):")
        buckets = pd.qcut(p, 10, duplicates="drop")
        table = pd.DataFrame({"p": p, "y": y.values}).groupby(buckets, observed=True).agg(
            n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
        )
        for _, r in table.iterrows():
            gap = r.predicted - r.observed
            print(f"    n={int(r.n):>6}  predicted {r.predicted:.3f}  observed {r.observed:.3f}  gap {gap:+.3f}")
        print()

    _calibration_check()


def _calibration_check() -> None:
    """Is the model miscalibrated, or is the holdout just a different period?"""
    from sklearn.model_selection import train_test_split

    df = load_frame()
    features = POST_DELIVERY_FEATURES

    def mean_gap(train, test):
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pipeline.fit(train[features], train["bad_review"])
        p = pipeline.predict_proba(test[features])[:, 1]
        y = test["bad_review"].values
        table = pd.DataFrame({"p": p, "y": y}).groupby(
            pd.qcut(p, 10, duplicates="drop"), observed=True
        ).agg(predicted=("p", "mean"), observed=("y", "mean"))
        from sklearn.metrics import roc_auc_score

        return (table.predicted - table.observed).mean(), roc_auc_score(y, p)

    split = int(len(df) * TRAIN_FRACTION)
    t_gap, t_auc = mean_gap(df.iloc[:split], df.iloc[split:])
    rtr, rte = train_test_split(df, test_size=1 - TRAIN_FRACTION, random_state=0, shuffle=True)
    r_gap, r_auc = mean_gap(rtr, rte)

    print("=== calibration ===")
    print(f"  temporal split   mean gap {t_gap:+.4f}   AUC {t_auc:.4f}")
    print(f"  random split     mean gap {r_gap:+.4f}   AUC {r_auc:.4f}")


if __name__ == "__main__":
    _report()
