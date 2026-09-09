"""The prediction path: "what review score will this order get?"

Parallel in shape to nl2sql.py, which holds the whole SQL path. This file holds
the whole prediction path - the feature query, the model, and the tool-call
extraction that turns a typed question into model inputs.

Two models are trained from one pipeline, and the difference between them is
the point:

  at_purchase   - only what is known when the order is placed. An honest
                  forecast, and a weak one.
  post_delivery - adds how long delivery actually took. Stronger, but it
                  consumes order_delivered_customer_date, which is not known
                  at order time. It is a WHAT-IF SIMULATOR, not a forecast:
                  the user supplies the delivery outcome as part of the
                  question, so the leak is the input.

Run `python predict.py` to print the evaluation report, including current
holdout AUC for both modes - not repeated as a fixed number here or in the
UI, because it changes whenever the feature set does and a hardcoded number
goes stale silently. See holdout_auc() below.
"""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The model call setup is shared with the SQL path rather than duplicated. The
# first version of this file redefined MODEL and its own Anthropic client to
# keep the two paths uncoupled, but the truncation fix needs the same token
# budget and the same exception type on both sides, and having two copies of
# "how we call the model" drift apart is a worse problem than the import.
from nl2sql import MAX_TOKENS, MODEL, TruncatedResponse, client

DB_PATH = Path(__file__).parent / "olist.duckdb"

# Fraction of orders (oldest first) used for training; the rest is holdout.
TRAIN_FRACTION = 0.8


# One query builds the entire training frame. Three things it has to get right:
#
#   1. 547 orders have more than one review. We take MIN(review_score) - the
#      worst review is the one a business cares about. A judgment call, not an
#      obvious truth: MAX or AVG would be equally defensible.
#   2. order_items and order_payments are one-row-per-item, so both are
#      collapsed to one row per order BEFORE joining, or every order with two
#      items would be counted twice.
#   3. Delivered orders only. This drops ~3,000 non-delivered orders whose
#      average review score is 1.3-2.0 - the worst outcomes in the dataset are
#      excluded from training by construction. See DESIGN_DECISIONS.md.
#
# The inner joins additionally drop 647 delivered orders (0.7%). Profiled by
# isolating each join: 646 are missing a review entirely, 1 is missing a
# payment, 0 are missing a resolvable seller - the review join does
# essentially all the dropping. Small enough to accept, large enough to
# mention.
FEATURE_SQL = """
WITH review AS (
    SELECT order_id, MIN(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
items AS (
    SELECT order_id,
           SUM(price)         AS price_total,
           SUM(freight_value) AS freight_total,
           COUNT(*)           AS n_items
    FROM order_items
    GROUP BY order_id
),
payments AS (
    SELECT order_id, MAX(payment_installments) AS installments
    FROM order_payments
    GROUP BY order_id
),
seller_states AS (
    SELECT oi.order_id,
           COUNT(DISTINCT s.seller_state) AS n_seller_states,
           MAX(s.seller_state)            AS a_seller_state
    FROM order_items oi
    JOIN sellers s ON s.seller_id = oi.seller_id
    GROUP BY oi.order_id
),
primary_seller AS (
    -- An order can have items from several sellers. The seller of the most
    -- expensive item is taken as "the" seller, so seller history has one
    -- value per order. Arbitrary for the rare multi-seller order (3% of them),
    -- but it has to collapse to one row somehow and "who supplied most of the
    -- order value" is the most defensible choice.
    SELECT order_id, seller_id FROM (
        SELECT oi.order_id, oi.seller_id,
               ROW_NUMBER() OVER (PARTITION BY oi.order_id ORDER BY oi.price DESC) AS rn
        FROM order_items oi
    ) WHERE rn = 1
)
SELECT
    o.order_purchase_timestamp,
    CAST(r.review_score <= 3 AS INTEGER) AS bad_review,
    DATE_DIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date)     AS promised_days,
    DATE_DIFF('day', o.order_purchase_timestamp, o.order_delivered_customer_date)     AS delivery_days,
    DATE_DIFF('day', o.order_estimated_delivery_date, o.order_delivered_customer_date) AS days_late,
    i.price_total,
    i.freight_total,
    i.n_items,
    p.installments,
    CAST(ss.n_seller_states = 1 AND ss.a_seller_state = c.customer_state AS INTEGER) AS same_state,
    ps.seller_id
FROM orders o
JOIN review         r  ON r.order_id    = o.order_id
JOIN customers      c  ON c.customer_id = o.customer_id
JOIN items          i  ON i.order_id    = o.order_id
JOIN payments       p  ON p.order_id    = o.order_id
JOIN seller_states  ss ON ss.order_id   = o.order_id
JOIN primary_seller ps ON ps.order_id   = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
"""

# Columns the model actually sees. Note these are the DERIVED names, not the
# raw ones - log_price/log_freight and the days_late hinge are built in
# _add_derived below.
AT_PURCHASE_FEATURES = [
    "promised_days",
    "log_price",
    "log_freight",
    "n_items",
    "installments",
    "same_state",
    "seller_hist_bad",
]
POST_DELIVERY_FEATURES = AT_PURCHASE_FEATURES + [
    "delivery_days",
    "is_late",
]

# Values a user can state in a question. These are the raw, human-facing names;
# everything in the lists above is derived from them.
#
# Split by mode, because the at_purchase model genuinely never sees the two
# delivery-outcome values. Reporting them as inputs would say the prediction
# used a delivery time it actually ignored - the precise thing the inputs table
# exists to stop the app doing.
AT_PURCHASE_USER_FEATURES = [
    "promised_days",
    "price_total",
    "freight_total",
    "n_items",
    "installments",
    "same_state",
    "seller_hist_bad",
]
POST_DELIVERY_USER_FEATURES = AT_PURCHASE_USER_FEATURES + ["delivery_days", "days_late"]
USER_FEATURES = POST_DELIVERY_USER_FEATURES

# Mentioned in questions, deliberately not model inputs. Kept in the extraction
# schema anyway so the app can say "you mentioned this and it was ignored"
# instead of silently dropping it.
IGNORED_FEATURES = ["product_category", "customer_state"]

# Plain-English meaning of each input, shown next to it in the UI. This is the
# only place that says what the features actually are, so it doubles as the
# list of what you can state in a question - most inputs end up defaulted to a
# median simply because the user didn't know they could name them. Units are
# spelled out because "freight 20" is ambiguous and "R$20" is not.
FEATURE_DESCRIPTIONS = {
    "promised_days": "Days from purchase to the promised delivery date",
    "price_total": "Total order value in reais (R$), all items",
    "freight_total": "Total shipping cost in reais (R$), all items",
    "n_items": "Number of items on the order",
    "installments": "Number of payment installments",
    "same_state": "1 if the seller is in the customer's state, 0 if not",
    "seller_hist_bad": "Seller's past bad-review rate, 0 to 1 (0.2 is typical)",
    "delivery_days": "Days the delivery actually took, purchase to arrival",
    "days_late": "Days later than promised - negative means it arrived early",
}

_frame_cache = None
_model_cache = {}


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the model is actually fit on.

    Two transformations, both there for a specific measured reason:

    log_price / log_freight - price_total is heavily skewed (median R$86, max
    R$13,440). Untransformed, a handful of huge orders dominate the fitted
    coefficient.

    is_late - THE important one. days_late is not linearly related to review
    score: it is a hinge. Everything below zero (91% of orders) sits flat at
    ~17-19% bad, then it jumps to 62% at one day late and 90% by eight days.
    Fed in as a single linear term, logistic regression spends one coefficient
    averaging over the huge early region where there is no signal, and holdout
    AUC comes out at 0.625. Replacing it with a plain "did it arrive late at
    all" indicator lets the model represent that jump directly, and AUC goes
    to 0.646.

    Note what is NOT here: a "how many days late" slope term. It was tried
    (days_late clipped at zero) and dropped. Because
    delivery_days = promised_days + days_late, adding it put three
    near-linearly-dependent terms in one linear model; the fit came back with
    a NEGATIVE coefficient on the slope, implying later deliveries are less
    likely to get a bad review, which the raw data flatly contradicts. That
    was a collinearity artifact, not a finding. Dropping it scores marginally
    better (0.6460 vs 0.6457) and costs nothing behaviourally - delivery_days
    still supplies a continuous gradient, so P(bad) rises smoothly from 0.24
    at on-time to 0.78 at 15 days late.

    What dropping it does NOT do: make every remaining coefficient positive.
    price, promised_days, and (in at_purchase) same_state come out small and
    negative - each independently explainable, not further collinearity - see
    DESIGN_DECISIONS.md ("Logistic regression, and what its coefficients
    can't tell you") for the measured table and why. The fix here removed the
    one coefficient that flatly contradicted the raw data; it didn't make
    every coefficient read as an obviously positive story.
    """
    df = df.copy()
    df["log_price"] = np.log1p(df["price_total"])
    df["log_freight"] = np.log1p(df["freight_total"])
    df["is_late"] = (df["days_late"] > 0).astype(int)
    return df


def _add_seller_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add seller_hist_bad: the share of this seller's EARLIER orders that got
    a bad review.

    Kept separate from _add_derived on purpose. _add_derived is pure row-wise
    arithmetic and also runs on a single hypothetical order at prediction time;
    this one needs the whole time-ordered history and only ever runs on the
    training frame. At prediction time the user either states a seller's track
    record or it falls back to a median, exactly like any other input.

    THE LEAKAGE TRAP, and why it is written this way. The obvious version -
    grouping by seller and taking the mean review score - lets every order see
    its own outcome, plus the outcomes of orders that had not happened yet.
    That scores beautifully and is worthless. `shift(1)` drops the current row
    and `expanding()` averages only the rows before it, so each order sees
    strictly its own past. The frame is sorted by purchase time before this
    runs, which is what makes "before it" mean "earlier in time".

    Unseen sellers (3.1% of training rows) get the training-set base rate, not
    a -1 sentinel. A sentinel works for a tree, which can split on it, but this
    is a linear model where -1 is simply a small number sitting below every
    real rate - it would read "brand new seller" as "outstanding seller".
    Measured: the base-rate fill scores 0.6638 against the sentinel's 0.6580.

    Honest caveat: reviews are written days after purchase, so a handful of
    those "earlier" reviews would not actually have existed yet at order time.
    This makes the feature very slightly optimistic. Fixing it properly means
    keying on review timestamps rather than purchase order - not worth it here,
    but it is the reason to treat the gain as an upper estimate.
    """
    df = df.copy()
    prior = df.groupby("seller_id")["bad_review"]
    df["seller_hist_bad"] = prior.transform(lambda s: s.shift(1).expanding().mean())

    # The fill value is computed from the training slice only, so the holdout
    # period contributes nothing to it.
    train_rate = df["bad_review"].iloc[: int(len(df) * TRAIN_FRACTION)].mean()
    df["seller_hist_bad"] = df["seller_hist_bad"].fillna(train_rate)
    return df


def load_frame() -> pd.DataFrame:
    """Run FEATURE_SQL and return the derived training frame, sorted oldest
    first so a temporal split is just a slice. Cached in a module global.

    Order matters: sort by time, then build seller history (which depends on
    that ordering), then the row-wise derived columns.
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
    """Return (fitted_pipeline, medians) for `mode`, training on first call.

    Trained on demand and cached in a module global rather than saved to a
    pickle - the same pattern validate.py uses for its schema cache. Fitting
    takes about a second on 77k rows, and Streamlit keeps modules imported
    across reruns, so the cost is paid once per session. Avoids a binary
    artifact that can silently go stale against the code that reads it.

    StandardScaler earns its place for readability, not accuracy. Measured on
    post_delivery: scaled and unscaled give the same holdout AUC (0.6638 vs
    0.6637 - the gap is float noise, not a real difference), so it changes no
    prediction. What it changes is convergence (9 lbfgs iterations scaled vs
    208 unscaled) and, more importantly, whether the coefficients can be
    compared to each other at all. Unscaled, is_late comes out at +1.76 and
    delivery_days at +0.043, which reads as "lateness matters 40x more" - but
    that is just the units talking, because is_late spans 0-1 and
    delivery_days spans 0-208. Scaled, they land close together (see the
    printed report). Since the whole argument for choosing logistic
    regression here was readable coefficients, scaling is what makes that
    argument true. This comparison is re-run by hand whenever the feature set
    changes rather than kept as a permanent fixture - it is here to make the
    point once, not to be a live metric (that is what holdout_auc() is for).
    """
    if mode not in _model_cache:
        features = _features_for(mode)
        df = load_frame()
        split = int(len(df) * TRAIN_FRACTION)
        train = df.iloc[:split]

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pipeline.fit(train[features], train["bad_review"])

        # Medians come from the TRAINING slice only, so a defaulted feature
        # never carries information from the holdout period.
        medians = train.median(numeric_only=True)
        _model_cache[mode] = (pipeline, medians)
    return _model_cache[mode]


def _features_for(mode: str) -> list[str]:
    return POST_DELIVERY_FEATURES if mode == "post_delivery" else AT_PURCHASE_FEATURES


_auc_cache = {}


def holdout_auc(mode: str) -> float:
    """Holdout ROC-AUC for `mode`, computed from the model rather than typed
    into a UI string by hand.

    This exists because of a bug: the app's captions used to hardcode
    "AUC 0.56" and "AUC 0.65" as text. When seller_hist_bad was added those
    numbers moved to 0.5886 and 0.6638, and the caption strings silently kept
    reporting the old ones - the exact kind of stale claim this whole project
    is built to avoid making about a query result. Any UI text that states a
    metric now calls this instead of embedding a number.
    """
    if mode not in _auc_cache:
        from sklearn.metrics import roc_auc_score

        pipeline, _ = get_model(mode)
        df = load_frame()
        test = df.iloc[int(len(df) * TRAIN_FRACTION) :]
        features = _features_for(mode)
        p = pipeline.predict_proba(test[features])[:, 1]
        _auc_cache[mode] = roc_auc_score(test["bad_review"], p)
    return _auc_cache[mode]


@dataclass
class Prediction:
    """Everything the UI needs to render one prediction honestly.

    `provided` vs `defaulted` is the field that matters: a question like "what
    if delivery is 10 days late?" states one value, and the other five are
    filled with dataset medians. Without showing that split, the app invents an
    order and reports a confident number about it.
    """

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

    Used to warn when a question asks about a scenario the model has barely
    seen. Logistic regression extrapolates a straight line forever: asked about
    a delivery 60 days late it happily returns 0.96, well outside anything in
    the data. The prediction is still returned - refusing would be worse - but
    the UI says which inputs were extreme so the number is read with the right
    amount of trust.
    """
    df = load_frame()
    train = df.iloc[: int(len(df) * TRAIN_FRACTION)]
    return {f: (train[f].quantile(0.01), train[f].quantile(0.99)) for f in USER_FEATURES}


def _resolve_inputs(stated: dict, medians) -> dict:
    """Fill in a complete feature row from whatever the question stated.

    The three delivery numbers are not independent:
    days_late = delivery_days - promised_days. A question usually gives one of
    them, so the other two are derived rather than defaulted, which keeps the
    row internally consistent. If a question somehow states both days_late and
    delivery_days, days_late wins and delivery_days is recomputed - it is the
    one the user is more likely to have meant literally.
    """
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
    """Score one hypothetical order.

    The mode picks itself: if the question supplied a delivery outcome we use
    the post_delivery model, otherwise the at_purchase one. That avoids asking
    the user to choose between two models they have no way to choose between,
    and it means the leaky model is only ever used when the user has explicitly
    supplied the leaky value.
    """
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

    # Report against this mode's inputs only, so nothing appears in the inputs
    # table that the model did not actually use.
    user_features = (
        POST_DELIVERY_USER_FEATURES if mode == "post_delivery" else AT_PURCHASE_USER_FEATURES
    )
    # Work out what was recomputed BEFORE deciding what counts as provided.
    # _resolve_inputs lets days_late win when a question states both it and
    # delivery_days, so in that case delivery_days is overwritten even though
    # the user did state it - asked about "a 10 day delivery that ran 3 days
    # late" the row ends up with delivery_days = 27. Marking it "from your
    # question" would attach the user's authority to a number we computed,
    # which is exactly the provenance claim the inputs table exists to make
    # honestly.
    derived = set()
    if stated.get("days_late") is not None:
        derived.add("delivery_days")
    elif stated.get("delivery_days") is not None:
        derived.add("days_late")

    provided = sorted(
        f for f in user_features if stated.get(f) is not None and f not in derived
    )
    defaulted = sorted(set(user_features) - set(provided) - derived)

    # Only worth flagging on values the user actually stated - a defaulted
    # median is by definition in range.
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
        # Deliberately no "required" list. "What if delivery is 10 days late?"
        # states exactly one value; requiring the rest would force the model to
        # invent an order rather than leave fields blank for us to default.
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
    separately so the UI can say they were dropped.

    Uses tool_choice to force the tool call. Without it the model answers
    conversationally on questions with nothing to extract, and we get no
    structured result at all - the point of this call is that its output shape
    is guaranteed, not that the model decides whether to use it.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=EXTRACT_SYSTEM,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": EXTRACT_TOOL["name"]},
        messages=[{"role": "user", "content": question}],
    )

    # A truncated response has no complete tool_use block, which is
    # indistinguishable from "the question contained nothing to extract" if you
    # only look at the payload. Without this check a perfectly specific
    # question comes back to the user as "no order details could be read from
    # that question", which sends them off rewording a question that was fine.
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


# Forcing tool_choice guarantees we get a structured result, but it also means
# the model MUST call the tool even when the question contains nothing to
# extract - and asked to fill a field it has no value for, it writes a filler
# like "<UNKNOWN>" or "N/A" rather than omitting it. Observed on "what about
# reviews?", which came back with customer_state="<UNKNOWN>" and produced a
# misleading "you mentioned customer_state" warning in the UI.
#
# Only the free-text fields do this; the numeric ones are correctly omitted,
# so this guard is deliberately limited to strings rather than being a general
# sanitiser. Same lesson as the repair loop in DESIGN_DECISIONS.md: constrain a
# model's output shape and it will satisfy the constraint with something.
_PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "null", "not specified", "not mentioned"}


def _is_placeholder(value) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().strip("<>").lower() in _PLACEHOLDERS


def answer_prediction(question: str) -> Prediction:
    """Question in, Prediction out. The prediction path's entry point,
    mirroring answer_question in nl2sql.py.

    A question with nothing extractable fails here rather than predicting on an
    all-median order. Returning a confident 21% for "what about reviews?" would
    be the app inventing a question it was never asked - the same failure mode
    DESIGN_DECISIONS.md already records for the SQL path with gibberish input.
    """
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
    """Is the model miscalibrated, or is the holdout just a different period?

    Every gap in the tables above is positive, which looks like overconfidence.
    It mostly is not. The holdout base rate is 0.166 against a training rate of
    0.222 - later orders simply got better reviews - so some positive gap is
    expected no matter how good the model is.

    Refitting the same model on a RANDOM split separates the two: same data,
    same features, no time drift. If the model is well calibrated in
    distribution, the random split's mean gap should sit near zero.
    """
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

    print("=== calibration: drift or overconfidence? (post_delivery) ===")
    print(f"  temporal split   mean gap {t_gap:+.4f}   AUC {t_auc:.4f}")
    print(f"  random split     mean gap {r_gap:+.4f}   AUC {r_auc:.4f}")
    print()
    print("  Read it this way: on the random split the mean gap is ~0, so logistic")
    print("  regression is already well calibrated here and does not need wrapping in")
    print("  CalibratedClassifierCV. The positive gaps on the temporal split are the")
    print("  base rate moving under the model, not the model being overconfident.")
    print("  The AUC difference is the other half of the lesson: a random split would")
    print("  have reported ~0.71 for a model that gets ~0.65 on genuinely unseen later")
    print("  orders. That gap is why the temporal split is the one being reported.")


if __name__ == "__main__":
    _report()
