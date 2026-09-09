"""Minimal Streamlit UI for the router and the two paths behind it."""

import time

import pandas as pd
import streamlit as st

from nl2sql import TruncatedResponse, answer_question, explain_result
from predict import FEATURE_DESCRIPTIONS, answer_prediction, holdout_auc
from router import classify
from tracing import new_record, write_turn

st.title("Talk With the Olist E-commerce Dataset ")

# 5 examples: 2 for the predict path, 2 for the sql path, and 1 that is unanswerable.
EXAMPLE_QUESTIONS = [
    "average review score by product category",
    "avg delivery time when seller is in the same state as customer",
    "a R$150 order with 2 items that arrives 10 days late",
    "what if a seller with a 30% bad review rate ships an order that arrives 5 days early?",
    "what is the average age of our customers?",
]


def use_example(text: str) -> None:
    st.session_state["question"] = text
    st.session_state["run"] = True



col_input, col_ask = st.columns([5, 1], vertical_alignment="bottom")

question = col_input.text_input(
    "Ask a question about the Olist e-commerce dataset",
    key="question",
    placeholder="e.g. what is the average delivery time for orders with 5 items?",
)


if col_ask.button("Ask", width="stretch"):
    st.session_state["run"] = True
st.caption(
    "Ask about what happened in the data, or describe a hypothetical order to "
    "get a predicted review score. The app works out which you meant."
)

st.write("Or try one of these:")
for example in EXAMPLE_QUESTIONS:
    st.button(example, key=f"example_{example}", on_click=use_example, args=(example,))


def render_prediction(prediction) -> None:
    """Show the prediction result in the UI, with explanations."""
    if prediction.ignored:
        mentioned = ", ".join(f"{k} = {v}" for k, v in prediction.ignored.items())
       
        st.warning(
            f"Ignored: {mentioned}. Product category and customer state are not "
            "inputs to this model. Both were measured against the feature set "
            "and added too little to justify the extra columns, so they were "
            "left out."
        )

    if prediction.error:
        st.error(prediction.error)
        return

    st.metric("Probability of a bad review (score 1-3)", f"{prediction.probability:.0%}")

    auc = holdout_auc(prediction.mode)
    if prediction.mode == "post_delivery":
        st.caption(
            f"**What-if mode.** This uses how long delivery actually took, which is "
            f"not known when an order is placed. Note that this is not a forecast. "
            f"Holdout AUC is {auc:.2f}. This determines how well the model can separate "
            f"good and bad reviews on unseen data, where 1.0 is perfect and 0.5 is random guessing."
        )
    else:
        st.caption(
            f"**At-purchase mode.** Only uses what is known when the order is placed. "
            f"This is a forecast, but note that holdout AUC is {auc:.2f}. "
            f"This determines how well the model can separate good and bad reviews "
            f"on unseen data, where 1.0 is perfect and 0.5 is random guessing. "
            f"The model is not very good at this, so treat the probability as directional only."
        )

    if prediction.out_of_range:
        st.warning(
            f"Outside the range of the training data: {', '.join(prediction.out_of_range)}. "
            "The model extrapolates a straight line beyond what it has seen, so treat "
            "this probability as directional only."
        )

    source = {}
    for feature in prediction.inputs:
        if feature in prediction.provided:
            source[feature] = "from your question"
        elif feature in prediction.defaulted:
            source[feature] = "dataset median"
        else:
            source[feature] = "derived"

    st.subheader("Inputs used")
    st.dataframe(
        pd.DataFrame(
            {
                "value": [round(v, 2) for v in prediction.inputs.values()],
                "source": [source[f] for f in prediction.inputs],
                "description": [FEATURE_DESCRIPTIONS[f] for f in prediction.inputs],
            },
            index=list(prediction.inputs),
        ),
        width="stretch",
    )
    st.caption(
        "Anything marked 'dataset median' can be set by saying so in the question. "
        "'derived' means the value follows from the other delivery values above "
        "(days_late = delivery_days - promised_days) - which may themselves be "
        "dataset medians, not only values you stated."
    )



if st.session_state.pop("run", False):
    started = time.perf_counter()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    record = new_record("unrouted", question)

    try:
        route = classify(question)
        record["mode"] = route.route
        record["route_reason"] = route.reason


        st.caption(f"Routed to **{route.route}** — {route.reason}".replace("$", r"\$"))

        if route.route == "predict":
            prediction = answer_prediction(question)
            render_prediction(prediction)
            record.update(
                pred_mode=prediction.mode,
                probability=prediction.probability,
                inputs=prediction.inputs,
                provided=prediction.provided,
                defaulted=prediction.defaulted,
                ignored=prediction.ignored,
                out_of_range=prediction.out_of_range,
                error=prediction.error,
            )

        elif route.route == "sql":
            answer = answer_question(question)

            st.subheader("Generated SQL")
            st.code(answer.sql, language="sql")
            explanation = None

            if answer.error is None:
                st.subheader("Result")
                st.dataframe(answer.df, height=400)
                if len(answer.df) > 20:
                    st.caption(
                        f"Showing all {len(answer.df)} rows. The explanation below "
                        "is based on the first 10 plus the 5 highest and 5 lowest, "
                        "not the full result."
                    )

                explanation = explain_result(question, answer.sql, answer.df)
                st.subheader("Answer")
                st.write(explanation.replace("$", r"\$"))
            else:
                st.error(answer.error)


            record.update(
                attempts=answer.attempts,
                final_sql=answer.sql,
                row_count=None if answer.df is None else len(answer.df),
                answer=explanation,
                error=answer.error,
            )


            if len(answer.attempts) > 1:
                with st.expander(
                    f"Query attempts ({len(answer.attempts)})",
                    expanded=answer.error is not None,
                ):
                    for attempt in answer.attempts:
                        st.markdown(f"**Attempt {attempt['n']}: {attempt['outcome']}**")
                        st.code(attempt["sql"], language="sql")
                        if attempt["detail"]:
                            st.caption(attempt["detail"])

        else:
            st.info(
                "I can't answer that from this dataset. I can query the Olist "
                "order data: orders, items, payments, reviews, customers, "
                "sellers and products. I can also predict how likely a hypothetical "
                "order is to get a bad review."
            )

    except TruncatedResponse as e:
        st.error(f"{e} Try a shorter question.")
        record["error"] = str(e)

    record["duration_s"] = round(time.perf_counter() - started, 1)
    write_turn(record)


    with st.expander("Trace (JSON)"):
        st.json(record)
