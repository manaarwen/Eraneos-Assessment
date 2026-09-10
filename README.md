# Eraneos Assessment

## How to Run

### Setup (one-time)
1. Create a `.env` file in the repo root and add your Claude API key:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```
2. Install dependencies and load data:
   ```bash
   pip install -r requirements.txt
   python load_data.py  
   ```

### Interactive UI
Ask questions about the Olist dataset or predict a bad user review (possible at purchase time or post delivery).
```bash
streamlit run app.py 
```

### Evaluate the Prediction Model
```bash
python predict.py
```

Prints holdout metrics for both models (`at_purchase` and `post_delivery`), their coefficients, reliability tables, and the temporal-vs-random split comparison.

## Short Implementation Overview and Corresponding Files

| Component | File | Purpose |
|-----------|------|---------|
| Data Loading | `load_data.py` | Loads CSV files from `archive/` into a DuckDB database. |
| Schema | `schema.py` | Describes the database schema to the LLM for context. |
| NL→SQL Core | `nl2sql.py` | Converts natural language questions to SQL queries using Claude. |
| Validation & Repair | `validate.py` | Validates generated SQL and repairs common errors with retry logic. |
| Router | `router.py` | Routes questions to the SQL pipeline or prediction model. |
| Prediction Model | `predict.py` | Predicts review scores at purchase or post-delivery stages. |
| UI | `app.py` | Streamlit interface for asking questions and viewing results. |
| Tracing | `tracing.py` | Logs all query attempts, errors, and retries for debugging. |
