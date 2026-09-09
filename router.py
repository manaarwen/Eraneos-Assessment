"""Question classification: which path should answer this question?
One model call in front of every turn. It labels the question `sql`, `predict`
or `unanswerable`, and app.py dispatches on that label. 
"""

from dataclasses import dataclass
from nl2sql import MAX_TOKENS, MODEL, TruncatedResponse, client
from schema import SCHEMA_BODY

ROUTE_TOOL = {
    "name": "route",
    "description": "Record which path should handle the user's question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": ["sql", "predict", "unanswerable"],
                "description": "Which path should handle this question.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One short sentence explaining the choice. When the route is "
                    "unanswerable this sentence is shown to the user, so say what "
                    "is missing rather than restating the question."
                ),
            },
        },
        "required": ["route", "reason"],
    },
}

ROUTE_SYSTEM = (
    SCHEMA_BODY
    + """
You do not write SQL. You classify the user's question into exactly one route
by calling the route tool.

sql
  A question about what is actually in the data above: counts, averages,
  totals, rankings, comparisons, trends over real historical orders.
  Examples: "how many orders were delivered late?", "average review score by
  product category", "which seller state ships fastest?"

predict
  A hypothetical or what-if order that does not exist in the data, where the
  user wants to know how likely it is to get a bad review. These describe order
  details, i.e. value, number of items, how late delivery is, how well rated the
  seller is, rather than asking for a fact about past orders.
  Examples: "a R$150 order with 2 items that arrives 10 days late",
  "what if a poorly rated seller ships 5 days late?"

unanswerable
  Anything else. This covers three things:
    - questions about data these tables do not contain (customer age, profit
      margin, marketing spend, anything not listed above),
    - general knowledge or chit-chat unrelated to this dataset,
    - input that is not a question at all, including gibberish.

The difference between sql and predict is real orders versus a hypothetical
one. "What is the average review score for late deliveries?" is sql, because it asks
about orders that happened. "How likely is a bad review if delivery is late?"
describing a specific hypothetical order is predict.

When a question is answerable from the tables above, prefer sql. Only choose
unanswerable when the data genuinely is not there, or the input is not a
question about this dataset.
"""
)


@dataclass
class Route:
    """Determine which path should answer a question, and why."""

    route: str
    reason: str


def classify(question: str) -> Route:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=ROUTE_SYSTEM,
        tools=[ROUTE_TOOL],
        tool_choice={"type": "tool", "name": ROUTE_TOOL["name"]},
        messages=[{"role": "user", "content": question}],
    )


    if response.stop_reason == "max_tokens":
        raise TruncatedResponse(f"Routing was cut off at max_tokens ({MAX_TOKENS}).")

    payload = None
    for block in response.content:
        if block.type == "tool_use":
            payload = block.input
            break


    if payload is None or "route" not in payload or "reason" not in payload:
        raise TruncatedResponse("Routing call did not return a classification.")

    route = Route(route=payload["route"], reason=payload["reason"])

    print(f"--- route: {route.route} ({route.reason}) ---")
    return route
