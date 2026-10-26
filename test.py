# # import pandas as pd
# # import numpy as np

# # from agent.agent import DumE


# # # -----------------------------------------
# # # Test dataset
# # # -----------------------------------------

# # df = pd.DataFrame({
# #     "Employee_ID": [
# #         "E001",
# #         "E002",
# #         "E003",
# #         "E004",
# #         "E005"
# #     ],

# #     "Age": [
# #         25,
# #         "twenty five",
# #         30,
# #         "N/A",
# #         40
# #     ],

# #     "Salary": [
# #         "₹42,000",
# #         "50000",
# #         "ERROR",
# #         "65000",
# #         np.nan
# #     ],

# #     "Gender": [
# #         "Female",
# #         "M",
# #         "female",
# #         "Male",
# #         "F"
# #     ],

# #     "Joining_Date": [
# #         "2025-09-05",
# #         "2024/01/10",
# #         "N/A",
# #         "garbage",
# #         "2023-12-01"
# #     ],

# #     "Department": [
# #         "Finance",
# #         "finance",
# #         "-",
# #         "IT",
# #         "Sales"
# #     ]
# # })


# # # -----------------------------------------
# # # Create Dum-E
# # # -----------------------------------------

# # dume = DumE(
# #     model="qwen3:4b"
# # )


# # # -----------------------------------------
# # # Run Dum-E
# # # -----------------------------------------

# # cleaned_df, schema = dume.clean(df)


# # # -----------------------------------------
# # # Results
# # # -----------------------------------------

# # print("\n" + "=" * 60)
# # print("ORIGINAL DATA")
# # print("=" * 60)

# # print(df)


# # print("\n" + "=" * 60)
# # print("DUM-E SCHEMA")
# # print("=" * 60)

# # print(schema)


# # print("\n" + "=" * 60)
# # print("CLEANED DATA")
# # print("=" * 60)

# # print(cleaned_df)


# # print("\n" + "=" * 60)
# # print("FINAL DTYPES")
# # print("=" * 60)

# # print(cleaned_df.dtypes)

# from pathlib import Path

# # Go one level up from Dum-E/ to your main parent folder
# MAIN_DIR = Path(__file__).resolve().parent.parent

# knowledge_dir = MAIN_DIR / "Knowledge"
# tools_dir = MAIN_DIR / "Tools"

# print("=== INSIDE KNOWLEDGE ===")
# for p in sorted(knowledge_dir.rglob("*")):
#     print(" ", p.relative_to(MAIN_DIR))

# print("\n=== INSIDE TOOLS ===")
# for p in sorted(tools_dir.rglob("*")):
#     if "__pycache__" not in str(p):
#         print(" ", p.relative_to(MAIN_DIR))

from local_rag import query_agent_knowledge

results = query_agent_knowledge("How should missing values in a numeric column be handled?", top_k=2)

for r in results:
    print(f"\n--- {r['source']} (Score: {r['similarity']}) ---")
    print(r["policy_text"] + "...")