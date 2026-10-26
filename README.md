# 🤖 Dum-E: Autonomous Data-Cleaning Agent

Dum-E is an autonomous, RAG-backed data-cleaning orchestrator designed to run locally on resource-constrained hardware (e.g., 4GB VRAM GPUs). It combines the reasoning capabilities of Small Language Models (SLMs like `qwen3:4b`) with robust, deterministic Python data transformation tools to automatically clean, normalize, and repair messy datasets.

## ✨ Key Features

* **Autonomous Tool Planning:** Dum-E acts as a true agent. It profiles the dataset to build a "Data Health Snapshot" and dynamically plans which repair tools to execute based on the data's exact needs.

* **Semantic Schema Inference:** Accurately infers column types (`id`, `numeric`, `categorical`, `datetime`, `text`) even when numbers are disguised as text (e.g., `₹ 3,942.08` or word-based numbers like `"two"`).

* **Local RAG Knowledge Base:** Dum-E's decision-making is guided by a local directory of Markdown policy files (e.g., Policy 06 for Categories, Policy 08 for Integrity). The RAG system handles context retrieval for missing value imputation and shorthand category mapping.

* **Deterministic Execution:** While the LLM acts as the "brain" for planning and decision-making, the actual transformations are executed by isolated, deterministic Python functions to ensure data integrity and avoid hallucinated data modifications.

* **Granular Audit Reporting:** Automatically generates a detailed `report.md` after every run, tracking before-and-after row counts, distinct category mappings (e.g., `"W" -> "west"`), cell-level changes, and a final Data Integrity Status (`pass`, `pass_with_warnings`, `fail`).

## 🧠 Architecture: The 3-Stage Pipeline

1. **Data Profiler (Stage 1):** Scans the incoming dataset (CSV/Parquet) and creates a compact profile of pandas dtypes, sample values, missing counts, and categorical distributions without blowing up the context window.

2. **The Agent Brain (Stage 2):** Queries the local RAG vector store for relevant cleaning policies. The LLM then outputs a strict JSON tool-execution plan, including targeted mapping dictionaries for categorical shorthand anomalies.

3. **Tool Executor & Validator (Stage 3):** Deterministically executes the planned toolchain:

   * `word_to_num`: Converts text numbers (e.g., "nine") to digits.
   * `numeric_cleaner`: Strips currency symbols, commas, and whitespace.
   * `datetime_cleaner`: Standardizes date strings into datetime objects.
   * `categorical_cleaner`: Normalizes casing, strips whitespace, and maps shorthand values based on the Agent's plan.
   * `missing_value_handler`: Applies LLM-chosen imputation (mean, median, mode) guided by skewness and RAG policy.
   * `dtype_converter`: Enforces final inferred schema types.

## 🛠️ Setup & Requirements

* **Python 3.10+**
* **Ollama** (running locally with `qwen3:4b` or your preferred SLM)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Er-Rajas/Dum-E.git
   cd Dum-E
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running and your target model is pulled:
   ```bash
   ollama pull qwen3:4b
   ```

## 🚀 Usage

Run Dum-E directly from your terminal by pointing it to your raw dataset and an output directory:

```bash
python agent/dume_runner.py "path/to/raw_data.csv" "output_directory/"
```

**Outputs Generated:**

* `cleaned_data.parquet`: The fully cleaned and typed dataset.
* `schema.json`: The final inferred schema mapping.
* `report.md`: A human-readable Markdown log of every decision, missing value imputation, and cell-level change Dum-E made.

## 📂 Project Structure

* `agent/`: Contains the main `agent.py` logic and the `dume_runner.py` CLI entry point.
* `Tools/Dum_E_tools/`: The suite of deterministic Python cleaning scripts (categorical, numeric, datetime, word_to_num, etc.).
* `Knowledge/Dum_E_knowledge/`: Markdown files acting as the local RAG knowledge base to dictate cleaning rules.

## 📝 Future Roadmap

* **Optimization:** Batch LLM prompts to reduce execution time.
* **Integration:** Wrap Dum-E into a Model Context Protocol (MCP) compatible tool so orchestrators like `FRIDAY` can trigger it seamlessly.