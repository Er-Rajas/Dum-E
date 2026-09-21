# Dum-E

**Project status:** Release candidate
**Version:** `1.0.0-rc`

Dum-E is a Python data-cleaning agent for pandas DataFrames. It uses a local Ollama model to infer the semantic type of each column, then applies deterministic cleaning tools and produces an auditable Markdown report.

## What it does

- Supports CSV.
- Infers each column as `id`, `numeric`, `categorical`, `datetime`, or `text`.
- Normalizes known missing-value representations such as `N/A`, `null`, `-`, `ERROR`, and `garbage`.
- Converts word numbers, cleans common numeric formats, normalizes categorical values, and parses dates.
- Flags unresolved invalid values as missing candidates and asks the user how to handle them.
- Enforces final pandas dtypes and writes an audit report containing the inferred schema, changes, validation results, and data preview.

## Data-analysis and cleaning flow

The pipeline deliberately distinguishes analysis from data mutation:

```text
Input dataset
  -> [Analysis] semantic type inference (LLM; column names, dtypes, 10-row sample)
  -> [Cleaning] missing-value normalization and deterministic repair tools
  -> [Analysis] value classification and invalid-value detection
  -> [Decision] user selects a treatment for missing values
  -> [Cleaning] dtype enforcement
  -> [Analysis] final validation and audit report
  -> Cleaned Parquet dataset + schema.json + report.md
```

The LLM determines the expected meaning of columns; it does not perform the transformations. Cleaning, validation, and reporting are handled by local deterministic code.

## Requirements

- Python 3.10 or later
- An Ollama installation with the selected model available locally
- Default model: `qwen3:4b`

Install the Python dependencies:

```bash
pip install -r requirments.txt
```

Pull the default model if needed:

```bash
ollama pull qwen3:4b
```

## Run from the command line

```bash
python agent/dume_runner.py <input_file> <output_folder>
```

Example:

```bash
python agent/dume_runner.py data/input.csv output
```

During a run, Dum-E may prompt for a missing-value method. Available methods are `ffill`, `bfill`, `mean`, `median`, `mode`, `drop`, and `increment`.

The output folder receives:

- `cleaned_data.parquet` — cleaned dataset
- `schema.json` — inferred semantic type for every column
- `report.md` — run audit report

For a small in-memory example, run:

```bash
python test.py
```

## Project layout

```text
agent/
  agent.py          Core Dum-E orchestration and reporting
  dume_runner.py    Command-line runner
tools/
  *_cleaner.py      Deterministic cleaning and conversion tools
  value_classifier.py
                     Post-cleaning value validation
  missing_*.py      Missing-value normalization and treatment
test.py             Example dataset and manual smoke test
```

## Planned improvements

1. Improve LLM response timing, including measuring inference latency and reducing prompt/token overhead.
2. Send only the data needed for an LLM decision—such as column metadata, representative samples, and invalid/missing-value summaries—instead of passing an entire DataFrame.
3. Make analysis stages explicit in code and reports so it is clear which results are inferred analysis, deterministic transformations, user decisions, and final validation.
4. Add automated tests for the individual tools and full CLI flow before the final `1.0.0` release.

## Current release notes

`1.0.0-rc` is a release candidate. Validate it with representative datasets before using it in production, especially where incorrect type inference or automatic missing-value treatment could affect downstream decisions.

## License

This project is distributed under the [Apache License 2.0](LICENSE).
