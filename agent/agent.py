import json
import re
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from ollama import chat

# Force UTF-8 on Windows so PowerShell Measure-Command never crashes on → or ✓
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    Dum-E V0 - autonomous tool-planning and RAG-backed data-cleaning agent.
    """

    ALLOWED_TYPES = {
        "id",
        "numeric",
        "categorical",
        "datetime",
        "text",
    }

    AVAILABLE_REPAIR_TOOLS = {
        "word_to_num": "Convert number words (e.g., 'two', 'nine') into numeric values.",
        "numeric_cleaner": "Strip currency symbols (₹, $, etc.), commas, and spaces from numeric columns.",
        "datetime_cleaner": "Parse and standardize date/time strings into datetime objects.",
        "categorical_cleaner": "Trim whitespace, standardize case, and apply audited shorthand value_map on categorical columns.",
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
        self.validation_status = "pass"

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
        extra_notes: str = "",
    ):
        changes = self._changed_cells(before, after)

        if not changes:
            desc = "No cell values changed."
            if extra_notes:
                desc = f"{extra_notes}\n\n{desc}"
            self.record(stage, desc)
            return

        lines = []
        if extra_notes:
            lines.extend([extra_notes, ""])

        lines.extend(
            [
                f"Changed cells: `{len(changes)}`",
                "",
                "| Column | Row | Before | After |",
                "|---|---:|---|---|",
            ]
        )

        for change in changes[:50]:
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
        duration = finished_at - started_at
        mins, secs = divmod(duration.total_seconds(), 60)

        orig_dups = int(original_df.duplicated().sum())
        clean_dups = int(cleaned_df.duplicated().sum())

        report = [
            "# Dum-E V0 Cleaning Report",
            "",
            "## Run Information",
            "",
            f"- **Model:** `{self.model}`",
            f"- **Validation Status:** `{self.validation_status}`",
            f"- **Started:** `{started_at.isoformat(timespec='seconds')}`",
            f"- **Finished:** `{finished_at.isoformat(timespec='seconds')}`",
            f"- **Duration:** `{int(mins)}m {secs:.2f}s`",
            f"- **Rows:** `{len(original_df)}` → `{len(cleaned_df)}`",
            f"- **Columns:** `{len(original_df.columns)}` → `{len(cleaned_df.columns)}`",
            f"- **Duplicate Rows:** `{orig_dups}` → `{clean_dups}`",
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

        self.log(f"✓ Report written to: {path} (Duration: {int(mins)}m {secs:.2f}s)")

    # ================================================================
    # LLM & RAG DECISION MAKING
    # ================================================================

    def ask_llm(self, prompt: str) -> str:
        # Prepend /no_think so Qwen3 outputs JSON directly without slow internal monologues
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "/no_think\n" + prompt,
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "num_ctx": 4096,
            },
        )
        return response["message"]["content"]

    @staticmethod
    def _parse_llm_json(raw: str, fallback: dict) -> dict:
        if not raw:
            return fallback

        # Strip any <think>...</think> blocks or markdown fences
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return fallback

    def infer_types(self, df: pd.DataFrame) -> dict:
        # Compact per-column profile so the LLM sees dirty values clearly without token bloat
        column_profiles = {}
        for col in df.columns:
            samples = [str(v) for v in df[col].dropna().unique()[:8]]
            column_profiles[col] = {
                "pandas_dtype": str(df[col].dtype),
                "samples": samples,
            }

        retrieved_chunks = query_agent_knowledge(
            "How to infer semantic column types id numeric categorical datetime text?",
            top_k=2,
        )
        policy_context = "\n\n".join(c["policy_text"] for c in retrieved_chunks)

        prompt = f"""You are Dum-E, an autonomous data cleaning agent.
Infer the intended semantic datatype of every column in the dataset.
Allowed types: id, numeric, categorical, datetime, text.

Important:
- Columns representing prices, quantities, sales, profits, ratings, discounts, or days are "numeric" even if they contain currency symbols (₹, $), commas, or number words ('two', 'nine').
- Identifier codes like 'O0001' are "id".
- Dates like '31-01-2026' are "datetime".
- Repeated group labels (regions, segments, categories, products, names, payment methods) are "categorical".

Policy Context:
{policy_context}

Column Profiles:
{json.dumps(column_profiles, indent=2)}

Return ONLY a JSON object mapping every column name to one allowed type string.
"""
        raw = self.ask_llm(prompt)
        parsed = self._parse_llm_json(raw, fallback={})

        schema = {}
        for col in df.columns:
            val = parsed.get(col)
            if isinstance(val, list):
                val = val[0] if val else None
            if isinstance(val, dict):
                val = val.get("type") or val.get("semantic_type")
            val = str(val).strip().lower() if val else ""

            if val not in self.ALLOWED_TYPES:
                # Deterministic safety check if LLM missed a column
                dtype_str = str(df[col].dtype)
                if "int" in dtype_str or "float" in dtype_str:
                    val = "numeric"
                elif "date" in dtype_str:
                    val = "datetime"
                else:
                    val = "text"
            schema[col] = val

        return schema

    # ================================================================
    # AGENTIC TOOL PLANNING & EXECUTION
    # ================================================================

    def plan_and_run_repair_tools(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [col for col, dtype in self.schema.items() if dtype == "numeric"]
        datetime_columns = [col for col, dtype in self.schema.items() if dtype == "datetime"]
        categorical_columns = [col for col, dtype in self.schema.items() if dtype == "categorical"]

        # Build a compact observation of categorical distributions so Dum-E spots shorthand like 'W' vs 'West'
        cat_distributions = {}
        for col in categorical_columns:
            vc = df[col].dropna().astype(str).str.strip().value_counts()
            if len(vc) <= 30:
                cat_distributions[col] = vc.to_dict()

        retrieved_chunks = query_agent_knowledge(
            "categorical normalization shorthand mapping case whitespace DUME-CATEGORY-001",
            top_k=2,
        )
        cat_policy = "\n\n".join(c["policy_text"] for c in retrieved_chunks)

        prompt = f"""You are Dum-E, an autonomous data-cleaning agent.
Inspect the dataset schema and categorical value distributions, then plan your tool execution.

Available Tools:
{json.dumps(self.AVAILABLE_REPAIR_TOOLS, indent=2)}

Inferred Schema:
{json.dumps(self.schema, indent=2)}

Categorical Value Counts (before normalization):
{json.dumps(cat_distributions, indent=2)}

Categorical Handling Policy:
{cat_policy}

Task:
1. Select which tools from Available Tools are needed for this dataset in execution order.
2. Inspect Categorical Value Counts for unambiguous single-letter or abbreviated shorthand labels that belong to an existing full category in the same column (for example, 'W' -> 'west' when 'West', 'East', 'North', 'South' exist in 'Region').
3. Return ONLY a JSON object with keys:
- "selected_tools": list of tool names to run
- "categorical_value_map": object mapping column name to a dictionary of {{"shorthand_lowercase": "canonical_lowercase"}} (only for unambiguous abbreviations in the same column)
- "plan_summary": 1-sentence explanation of the chosen tools and mappings
"""
        raw = self.ask_llm(prompt)
        plan = self._parse_llm_json(raw, fallback={})

        selected_tools = plan.get(
            "selected_tools",
            ["word_to_num", "numeric_cleaner", "datetime_cleaner", "categorical_cleaner"],
        )
        if not isinstance(selected_tools, list) or not selected_tools:
            selected_tools = ["word_to_num", "numeric_cleaner", "datetime_cleaner", "categorical_cleaner"]

        cat_value_map = plan.get("categorical_value_map", {})
        if not isinstance(cat_value_map, dict):
            cat_value_map = {}

        plan_summary = plan.get("plan_summary", "Executed standard deterministic repair toolkit.")
        self.log(f"\n[Agent Plan] {plan_summary}")
        self.log(f"[Agent Tools] {', '.join(selected_tools)}")
        if cat_value_map:
            self.log(f"[Agent Category Map] {json.dumps(cat_value_map)}")

        self.record(
            "Agent Tool Planning",
            f"- **Selected Tools:** `{', '.join(selected_tools)}`\n"
            f"- **Categorical Shorthand Map:** `{json.dumps(cat_value_map)}`\n"
            f"- **Reasoning:** {plan_summary}",
        )

        # Execute Tool 1: word_to_num
        if "word_to_num" in selected_tools or numeric_columns:
            self.log("\n[3] Executing Tool: word_to_num...")
            before = df.copy()
            df = word_to_num(df)
            self._record_dataframe_changes("Word Number Conversion (`word_to_num`)", before, df)

        # Execute Tool 2: numeric_cleaner
        if ("numeric_cleaner" in selected_tools or numeric_columns) and numeric_columns:
            self.log("\n[4] Executing Tool: numeric_cleaner...")
            before = df.copy()
            df = numeric_cleaner(df, numeric_columns)
            self._record_dataframe_changes("Numeric Cleaning (`numeric_cleaner`)", before, df)

        # Execute Tool 3: datetime_cleaner
        if ("datetime_cleaner" in selected_tools or datetime_columns) and datetime_columns:
            self.log("\n[5] Executing Tool: datetime_cleaner...")
            before = df.copy()
            df = datetime_cleaner(df, datetime_columns)
            self._record_dataframe_changes("Datetime Cleaning (`datetime_cleaner`)", before, df)

        # Execute Tool 4: categorical_cleaner (with Policy 06 audit metrics)
        if ("categorical_cleaner" in selected_tools or categorical_columns) and categorical_columns:
            self.log("\n[6] Executing Tool: categorical_cleaner...")
            before = df.copy()
            distinct_before = {col: int(before[col].dropna().nunique()) for col in categorical_columns}

            df = categorical_cleaner(df, categorical_columns, value_map=cat_value_map)

            distinct_after = {col: int(df[col].dropna().nunique()) for col in categorical_columns}
            audit_lines = [
                "**Policy 06 Category Audit (Distinct Counts Before → After):**",
                "",
                "| Column | Distinct Before | Distinct After | Applied Mapping |",
                "|---|---:|---:|---|",
            ]
            for col in categorical_columns:
                applied = json.dumps(cat_value_map.get(col, {})) if col in cat_value_map else "None"
                audit_lines.append(
                    f"| `{col}` | `{distinct_before[col]}` | `{distinct_after[col]}` | `{applied}` |"
                )

            self._record_dataframe_changes(
                "Categorical Cleaning (`categorical_cleaner`)",
                before,
                df,
                extra_notes="\n".join(audit_lines),
            )

        return df

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
            self.validation_status = "pass_with_warnings"
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

            # 2. Determine allowed methods based on semantic & pandas dtype
            is_num = (
                semantic_type == "numeric"
                or pd.api.types.is_numeric_dtype(df[column])
            )
            if is_num:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                allowed = sorted(AVAILABLE_METHODS)
            else:
                allowed = sorted(AVAILABLE_METHODS - {"mean", "median"})

            # Calculate skewness hint for numeric columns so RAG policy chooses mean vs median accurately
            skew_val = float(df[column].skew()) if is_num and df[column].dropna().count() > 2 else 0.0

            # 3. Ask Qwen to select the method based on policy
            prompt = f"""You are Dum-E, a data cleaning agent. Select the best missing-value handling method for column '{column}'.

Column type: {semantic_type}
Missing count: {before_missing}/{total} ({percentage:.2f}%)
Skewness (if numeric): {skew_val:.2f}
Allowed methods: {allowed}

Retrieved Policies:
{policy_context}

Return ONLY a JSON object with keys:
"method": one string from allowed methods,
"reason": a brief 1-sentence justification citing the policy.
"""
            raw = self.ask_llm(prompt)
            decision = self._parse_llm_json(raw, fallback={})
            method = str(decision.get("method", "")).strip().lower()

            if method not in allowed:
                method = ("median" if abs(skew_val) > 1.0 else "mean") if is_num else "mode"
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
            final_types[column] = mapping.get(semantic_type, "text")

        before = df.copy()
        result = dtype_converter(df, final_types)

        dtype_changes = []
        for column in result.columns:
            old_dtype = str(before[column].dtype)
            new_dtype = str(result[column].dtype)
            if old_dtype != new_dtype:
                dtype_changes.append(f"- `{column}`: `{old_dtype}` → `{new_dtype}`")

        self.record(
            "Final Dtype Conversion",
            "\n".join(dtype_changes) if dtype_changes else "No dtype changes required.",
        )
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
        self.validation_status = "pass"

        started_at = datetime.now()

        self.log("\n" + "=" * 60)
        self.log("DUM-E V0 (AUTONOMOUS RAG & TOOL-PLANNING MODE)")
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

            # Agent plans and executes repair tools based on data inspection + RAG policies
            working_df = self.plan_and_run_repair_tools(working_df)

            self.log("\n[7] Re-classifying values after cleaning...")
            classification_report = self.classify(working_df)

            self.log("\n[8] Resolving remaining invalid values...")
            working_df = self.promote_invalid_to_missing(working_df, classification_report)

            self.log("\n[9] Autonomous RAG missing value handling...")
            classification_report = self.classify(working_df)
            working_df = self.handle_missing_values(working_df, classification_report)

            working_df = self.final_dtype_conversion(working_df)

            # Policy 08 Integrity Check
            if len(working_df) != len(original_df) or list(working_df.columns) != list(original_df.columns):
                self.validation_status = "fail"
            elif working_df.isna().sum().sum() > 0 and self.validation_status == "pass":
                self.validation_status = "pass_with_warnings"

            finished_at = datetime.now()
            self.write_report(original_df, working_df, started_at, finished_at)

            self.log("\n" + "=" * 60)
            self.log("DUM-E CLEANING COMPLETE")
            self.log("=" * 60)

            return working_df, self.schema

        except Exception as exc:
            self.validation_status = "fail"
            finished_at = datetime.now()
            self.record("Run Failed", f"`{type(exc).__name__}`: {exc}")
            try:
                self.write_report(original_df, working_df, started_at, finished_at)
            except Exception:
                pass
            raise