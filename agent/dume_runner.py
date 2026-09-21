import sys
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.agent import DumE


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_data(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(input_path)

    if suffix == ".parquet":
        return pd.read_parquet(input_path)

    raise ValueError(
        f"Unsupported input format: {suffix}. "
        "Use CSV, XLSX, XLS or Parquet."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python dume_runner.py <input_file> <output_folder>"
        )
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()
    output_folder = Path(sys.argv[2]).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    print("=" * 60)
    print("DUM-E STARTING")
    print("=" * 60)

    print(f"Input : {input_file}")
    print(f"Output: {output_folder}")

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("\n[1/4] Loading dataset...")

    df = load_data(input_file)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------
    # Run Dum-E
    # --------------------------------------------------

    print("\n[2/4] Running Dum-E...")

    dume = DumE(model="qwen3:4b")

    cleaned_df, schema = dume.clean(df)

    # --------------------------------------------------
    # Save cleaned dataframe
    # --------------------------------------------------

    print("\n[3/4] Saving cleaned dataset...")

    parquet_path = output_folder / "cleaned_data.parquet"

    cleaned_df.to_parquet(
        parquet_path,
        index=False
    )

    # --------------------------------------------------
    # Save schema
    # --------------------------------------------------

    schema_path = output_folder / "schema.json"

    with open(
        schema_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            schema,
            f,
            indent=4,
            default=str
        )

    # --------------------------------------------------
    # Check report
    # --------------------------------------------------

    report_path = output_folder / "report.md"

    # Dum-E should already generate this.
    # If it exists in the project root, copy it.
    project_report = PROJECT_ROOT / "report.md"

    if project_report.exists() and project_report.resolve() != report_path.resolve():
        report_path.write_text(
            project_report.read_text(encoding="utf-8"),
            encoding="utf-8"
        )

    # --------------------------------------------------
    # Completion
    # --------------------------------------------------

    print("\n[4/4] DUM-E COMPLETE")

    print(f"\nCleaned data : {parquet_path}")
    print(f"Schema       : {schema_path}")
    print(f"Report       : {report_path}")

    print("\nDUME_COMPLETE")


if __name__ == "__main__":
    main()