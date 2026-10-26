import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from ollama import chat

# Setup paths to locate local RAG and tools correctly
AGENT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag import query_agent_knowledge

from Tools.Dum_E_tools.missing_normalization import bad_values
from Tools.Dum_E_tools.value_classifier import classify_values
from Tools.Dum_E_tools.word_to_num import word_to_num
from Tools.Dum_E_tools.numeric_cleaner import numeric_cleaner
from Tools.Dum_E_tools.datetime_cleaner import datetime_cleaner
from Tools.Dum_E_tools.categorical_cleaner import categorical_cleaner
from Tools.Dum_E_tools.dtype_converter import dtype_converter
from Tools.Dum_E_tools.missing_value_handler import (
    missing_value_handler,
    AVAILABLE_METHODS,
)


DEFAULT_MODEL = "qwen3:4b"


class DumE:
    """
    Dum-E V0 - autonomous deterministic data-cleaning orchestrator.

    Qwen3 is used for semantic column-type inference and RAG-backed
    missing value decision making. Actual transformations are deterministic.
    A Markdown execution report is generated automatically after each run.
    """

    ALLOWED_TYPES = {
        "id",
        "numeric",
        "categorical",
        "datetime",
        "text",
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        report_path: str = "report.md",
    ):
        self.model = model
        self.report_path = report_path
        self.schema = {}
        self.events = []

    # ================================================================
    # REPORT / LOGGING
    # ================================================================

    def log(self, message: str):
        """Print a message to the terminal."""
        print(message)

    def record(self, stage: str, description: str):
        """Record a structured event for report.md."""
        self.events.append(
            {
                "stage": stage,
                "description": description,
            }
        )

    @staticmethod
    def _changed_cells(before: pd.DataFrame, after: pd.DataFrame):
        changes = []
        common_columns = [
            col for col in before.columns
            if col in after.columns
        ]

        for column in common_columns:
            before_col = before[column]
            after_col = after[column]

            for index in before.index.intersection(after.index):
                old = before_col.loc[index]
                new = after_col.loc[index]

                old_missing = pd.isna(old)
                new_missing = pd.isna(new)

                if old_missing and new_missing:
                    continue

                if old_missing != new_missing or old != new:
                    changes.append(
                        {
                            "column": column,
                            "index": index,
                            "old": old,
                            "new": new,
                        }
                    )

        return changes

    def _record_dataframe_changes(
        self,
        stage: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ):
        changes = self._changed_cells(before, after)

        if not changes:
            self.record(stage, "No cell values changed.")
            return

        lines = [
            f"Changed cells: `{len(changes)}`",
            "",
            "| Column | Row | Before | After |",
            "|---|---:|---|---|",
        ]

        display_changes = changes[:50]

        for change in display_changes:
            lines.append(
                f"| `{change['column']}` | `{change['index']}` | "
                f"`{change['old']}` | `{change['new']}` |"
            )

        if len(changes) > 50:
            lines.extend(
                [
                    "",
                    f"> Only the first 50 of {len(changes)} changes are shown.",
                ]
            )

        self.record(stage, "\n".join(lines))

    def write_report(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        started_at: datetime,
        finished_at: datetime,
    ):
        report = [
            "# Dum-E V0 Cleaning Report",
            "",
            "## Run Information",
            "",
            f"- **Model:** `{self.model}`",
            f"- **Started:** `{started_at.isoformat(timespec='seconds')}`",
            f"- **Finished:** `{finished_at.isoformat(timespec='seconds')}`",
            f"- **Rows:** `{len(original_df)}` → `{len(cleaned_df)}`",
            f"- **Columns:** `{len(original_df.columns)}` → `{len(cleaned_df.columns)}`",
            "",
            "## Inferred Schema",
            "",
            "| Column | Semantic Type |",
            "|---|---|",
        ]

        for column, dtype in self.schema.items():
            report.append(f"| `{column}` | `{dtype}` |")

        report.extend(
            [
                "",
                "## Changes Made",
                "",
            ]
        )

        for number, event in enumerate(self.events, start=1):
            report.extend(
                [
                    f"### {number}. {event['stage']}",
                    "",
                    event["description"],
                    "",
                ]
            )

        report.extend(
            [
                "## Final Data Types",
                "",
                "| Column | dtype |",
                "|---|---|",
            ]
        )

        for column, dtype in cleaned_df.dtypes.items():
            report.append(f"| `{column}` | `{dtype}` |")

        report.extend(
            [
                "",
                "## Remaining Missing Values",
                "",
                "| Column | Missing Count | Percentage |",
                "|---|---:|---:|",
            ]
        )

        total_rows = len(cleaned_df)

        for column in cleaned_df.columns:
            count = int(cleaned_df[column].isna().sum())
            percentage = (count / total_rows * 100) if total_rows else 0
            report.append(
                f"| `{column}` | `{count}` | `{percentage:.2f}%` |"
            )

        report.extend(
            [
                "",
                "## Final Dataset Preview",
                "",
                "```text",
                cleaned_df.head(10).to_string(),
                "```",
                "",
                "---",
                "*Report generated autonomously by Dum-E via Local RAG.*",
                "",
            ]
        )

        path = Path(self.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report), encoding="utf-8")

        self.log(f"✓ Report written to: {path}")

    # ================================================================
    # LLM & RAG DECISION MAKING
    # ================================================================

    def ask_llm(self, prompt: str) -> str:
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format="json",
            options={
                "temperature": 0,
            },
        )
        return response["message"]["content"]

    def infer_types(self, df: pd.DataFrame) -> dict:
        sample = df.head(10).to_dict(orient="records")
        pandas_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        prompt = f"""
You are Dum-E, a data cleaning agent. Infer the intended semantic datatype of every dataframe column.
Allowed types: id, numeric, categorical, datetime, text.
Return ONLY a JSON object mapping column names to one allowed type.

Pandas dtypes:
{json.dumps(pandas_dtypes, indent=2)}

First 10 rows:
{json.dumps(sample, default=str, indent=2)}
"""
        raw = self.ask_llm(prompt)
        schema = json.loads(raw)
        return schema

    def classify(self, df: pd.DataFrame):
        return classify_values(df, self.schema)

    def promote_invalid_to_missing(
        self,
        df: pd.DataFrame,
        classification_report: dict,
    ) -> pd.DataFrame:
        df = df.copy()
        changes = []

        for column, report in classification_report.items():
            invalid_indexes = report.get("invalid", [])
            for index in invalid_indexes:
                old_value = df.at[index, column]
                df.at[index, column] = pd.NA
                changes.append(
                    {
                        "column": column,
                        "index": index,
                        "old": old_value,
                    }
                )

        if changes:
            lines = [f"Invalid values promoted to missing candidates: `{len(changes)}`", ""]
            for change in changes[:50]:
                lines.append(f"- `{change['column']}` row `{change['index']}`: `{change['old']}` → `NaN`")
            self.record("Invalid Value Resolution", "\n".join(lines))
        else:
            self.record("Invalid Value Resolution", "No unresolved invalid values found.")

        return df

    # ================================================================
    # AUTONOMOUS RAG MISSING VALUE HANDLING
    # ================================================================

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        classification_report: dict,
    ) -> pd.DataFrame:
        columns_with_missing = [
            column for column in df.columns if df[column].isna().any()
        ]

        if not columns_with_missing:
            self.log("✓ No missing values require imputation.")
            self.record(
                "Missing Value Handling",
                "No missing values were present after normalization and cleaning.",
            )
            return df

        self.log("\n" + "=" * 60)
        self.log("AUTONOMOUS RAG MISSING VALUE HANDLING")
        self.log("=" * 60)

        for column in columns_with_missing:
            total = len(df)
            before_missing = int(df[column].isna().sum())
            if before_missing == 0:
                continue

            percentage = (before_missing / total * 100) if total else 0.0
            semantic_type = self.schema.get(column, "text")

            # 1. Query Local RAG knowledge base for rules
            rag_query = f"How to handle {percentage:.1f}% missing values in a {semantic_type} column?"
            retrieved_chunks = query_agent_knowledge(rag_query, top_k=2)
            policy_context = "\n\n".join(c["policy_text"] for c in retrieved_chunks)

            # 2. Determine allowed methods based on column data type
            if pd.api.types.is_numeric_dtype(df[column]):
                allowed = sorted(AVAILABLE_METHODS)
            else:
                allowed = sorted(AVAILABLE_METHODS - {"mean", "median"})

            # 3. Ask Qwen to select the method based on policy
            prompt = f"""
You are Dum-E, a data cleaning agent. Select the best missing-value handling method for column '{column}'.

Column type: {semantic_type}
Missing count: {before_missing}/{total} ({percentage:.2f}%)
Allowed methods: {allowed}

Retrieved Policies:
{policy_context}

Return ONLY a JSON object with keys:
"method": one string from allowed methods,
"reason": a brief 1-sentence justification citing the policy.
"""
            try:
                raw = self.ask_llm(prompt)
                decision = json.loads(raw)
                method = str(decision.get("method", "")).strip().lower()
            except Exception:
                method = ""

            if method not in allowed:
                method = "median" if pd.api.types.is_numeric_dtype(df[column]) else "mode"
                reason = "Fallback chosen due to parser validation."
            else:
                reason = decision.get("reason", "Selected via RAG policy.")

            # 4. Apply handling method
            df = missing_value_handler(df, column, method)
            after_missing = int(df[column].isna().sum())

            self.record(
                f"Missing Value Handling - `{column}`",
                f"- **Semantic type:** `{semantic_type}`\n"
                f"- **Missing before:** `{before_missing}/{total}` (`{percentage:.2f}%`)\n"
                f"- **Selected method:** `{method}`\n"
                f"- **Justification:** {reason}\n"
                f"- **Missing after:** `{after_missing}`",
            )

            self.log(f"✓ {column:<20} → {method:<10} ({reason})")

        return df

    # ================================================================
    # REPAIR & DTYPE TOOLS
    # ================================================================

    def run_repair_tools(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [col for col, dtype in self.schema.items() if dtype == "numeric"]
        datetime_columns = [col for col, dtype in self.schema.items() if dtype == "datetime"]
        categorical_columns = [col for col, dtype in self.schema.items() if dtype == "categorical"]

        self.log("\n[3] Converting word-based numbers...")
        before = df.copy()
        df = word_to_num(df)
        self._record_dataframe_changes("Word Number Conversion", before, df)

        self.log("\n[4] Cleaning numeric representations...")
        before = df.copy()
        if numeric_columns:
            df = numeric_cleaner(df, numeric_columns)
        self._record_dataframe_changes("Numeric Cleaning", before, df)

        self.log("\n[5] Cleaning datetime representations...")
        before = df.copy()
        if datetime_columns:
            df = datetime_cleaner(df, datetime_columns)
        self._record_dataframe_changes("Datetime Cleaning", before, df)

        self.log("\n[6] Normalizing categorical values...")
        before = df.copy()
        if categorical_columns:
            df = categorical_cleaner(df, categorical_columns)
        self._record_dataframe_changes("Categorical Cleaning", before, df)

        return df

    def final_dtype_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log("\n[10] Enforcing final dtypes...")
        final_types = {}
        for column, semantic_type in self.schema.items():
            mapping = {
                "id": "id",
                "numeric": "numeric",
                "datetime": "datetime",
                "categorical": "category",
                "text": "text",
            }
            final_types[column] = mapping[semantic_type]

        before = df.copy()
        result = dtype_converter(df, final_types)
        
        dtype_changes = []
        for column in result.columns:
            old_dtype = str(before[column].dtype)
            new_dtype = str(result[column].dtype)
            if old_dtype != new_dtype:
                dtype_changes.append(f"- `{column}`: `{old_dtype}` → `{new_dtype}`")

        self.record("Final Dtype Conversion", "\n".join(dtype_changes) if dtype_changes else "No dtype changes required.")
        return result

    # ================================================================
    # MAIN PIPELINE
    # ================================================================

    def clean(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Dum-E expects a pandas DataFrame.")

        original_df = df.copy(deep=True)
        working_df = df.copy(deep=True)
        self.schema = {}
        self.events = []

        started_at = datetime.now()

        self.log("\n" + "=" * 60)
        self.log("DUM-E V0 (AUTONOMOUS RAG MODE)")
        self.log("=" * 60)

        try:
            self.log("\n[1] Inferring semantic column types...")
            self.schema = self.infer_types(working_df)
            for column, dtype in self.schema.items():
                self.log(f"  {column:<25} → {dtype}")

            self.log("\n[2] Normalizing known missing representations...")
            before = working_df.copy()
            working_df = bad_values(working_df)
            self._record_dataframe_changes("Missing Representation Normalization", before, working_df)

            working_df = self.run_repair_tools(working_df)

            self.log("\n[7] Re-classifying values after cleaning...")
            classification_report = self.classify(working_df)

            self.log("\n[8] Resolving remaining invalid values...")
            working_df = self.promote_invalid_to_missing(working_df, classification_report)

            self.log("\n[9] Autonomous RAG missing value handling...")
            classification_report = self.classify(working_df)
            working_df = self.handle_missing_values(working_df, classification_report)

            working_df = self.final_dtype_conversion(working_df)

            finished_at = datetime.now()
            self.write_report(original_df, working_df, started_at, finished_at)

            self.log("\n" + "=" * 60)
            self.log("DUM-E CLEANING COMPLETE")
            self.log("=" * 60)

            return working_df, self.schema

        except Exception as exc:
            finished_at = datetime.now()
            self.record("Run Failed", f"`{type(exc).__name__}`: {exc}")
            try:
                self.write_report(original_df, working_df, started_at, finished_at)
            except Exception:
                pass
            raise