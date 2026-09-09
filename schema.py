"""Description of the DuckDB schema, fed to the model as part of
the system prompt in nl2sql.py and router.py."""

SCHEMA_BODY = """
You are querying a DuckDB database with eight tables from the Olist Brazilian
e-commerce dataset.

TABLE orders
  order_id                       VARCHAR   primary key
  customer_id                    VARCHAR   foreign key -> customers.customer_id
  order_status                   VARCHAR   one of: delivered, shipped, canceled,
                                            unavailable, invoiced, processing,
                                            created, approved
  order_purchase_timestamp       TIMESTAMP when the order was placed
  order_approved_at              TIMESTAMP when payment was approved
  order_delivered_carrier_date   TIMESTAMP when handed to the carrier
  order_delivered_customer_date  TIMESTAMP when the customer actually received it
  order_estimated_delivery_date  TIMESTAMP the delivery date promised at purchase

TABLE order_reviews
  review_id                 VARCHAR   primary key
  order_id                  VARCHAR   foreign key -> orders.order_id
  review_score              BIGINT    1 to 5, 5 is best
  review_comment_title      VARCHAR   free text, often empty
  review_comment_message    VARCHAR   free text, often empty
  review_creation_date      TIMESTAMP
  review_answer_timestamp   TIMESTAMP

TABLE customers
  customer_id                VARCHAR   primary key, one row per ORDER
  customer_unique_id         VARCHAR   identifies the actual PERSON
  customer_zip_code_prefix   VARCHAR
  customer_city              VARCHAR
  customer_state             VARCHAR   two-letter Brazilian state code

TABLE order_items
  order_id               VARCHAR   foreign key -> orders.order_id
  order_item_id           BIGINT   line number within the order, starts at 1
  product_id              VARCHAR  foreign key -> products.product_id
  seller_id                VARCHAR
  shipping_limit_date      TIMESTAMP
  price                    DOUBLE   price of this one item, not the whole order
  freight_value             DOUBLE  shipping cost for this one item

TABLE products
  product_id                    VARCHAR   primary key
  product_category_name         VARCHAR   in PORTUGUESE, foreign key ->
                                           product_category_translation.product_category_name
  product_name_lenght           BIGINT
  product_description_lenght    BIGINT
  product_photos_qty            BIGINT
  product_weight_g               BIGINT
  product_length_cm              BIGINT
  product_height_cm              BIGINT
  product_width_cm               BIGINT

TABLE product_category_translation
  product_category_name           VARCHAR   Portuguese name, joins to products
  product_category_name_english   VARCHAR   English name for display

TABLE order_payments
  order_id                VARCHAR   foreign key -> orders.order_id
  payment_sequential       BIGINT   order of this payment when an order used
                                     more than one (e.g. voucher + credit card)
  payment_type              VARCHAR one of: credit_card, boleto, voucher,
                                     debit_card, not_defined
  payment_installments      BIGINT  number of installments for this payment
  payment_value              DOUBLE amount charged for this one payment

TABLE sellers
  seller_id                VARCHAR   primary key, foreign key from
                                      order_items.seller_id
  seller_zip_code_prefix   VARCHAR
  seller_city              VARCHAR
  seller_state             VARCHAR   two-letter Brazilian state code

IMPORTANT SEMANTIC NOTES

- customer_id vs customer_unique_id: customer_id is generated fresh for each
  order, so counting distinct customer_id counts ORDERS, not people. To count
  actual customers, or to find repeat customers, use customer_unique_id.

- Delivery timestamps are only reliably populated when order_status =
  'delivered'. For other statuses, order_delivered_customer_date is often
  NULL. Any question about delivery time or lateness should filter to
  order_status = 'delivered' first.

- "Late delivery" means order_delivered_customer_date is AFTER
  order_estimated_delivery_date. Do not confuse this with comparing against
  order_approved_at or order_purchase_timestamp.

- order_items has one row per item, not one row per order. An order with
  three items produces three rows with the same order_id. To get an order's
  total value, sum price (and usually freight_value) grouped by order_id.

- order_payments has one row per payment, not one row per order. An order
  paid with two methods (e.g. voucher + credit card) produces two rows with
  the same order_id. To get what an order actually cost, sum payment_value
  grouped by order_id - this is the authoritative order total, more reliable
  than summing order_items price + freight_value, since it reflects what was
  actually charged rather than list price.

- product_category_name in the products table is in Portuguese
  (e.g. 'beleza_saude', 'informatica_acessorios'). Join
  product_category_translation to get the English name
  (product_category_name_english) for anything shown to the user.

- review_score is an integer 1 to 5. There is no "average" row; compute
  averages with AVG(review_score).
"""

SQL_ONLY_INSTRUCTION = """
Return only a single DuckDB-compatible SQL query that answers the question.
Do not include any explanation, markdown formatting, or commentary, only the
SQL statement itself.
"""

SCHEMA_DESCRIPTION = SCHEMA_BODY + SQL_ONLY_INSTRUCTION
