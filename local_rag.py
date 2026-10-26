import json
from pathlib import Path
import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
MAIN_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = MAIN_DIR / "Knowledge" / "Dum_E_knowledge"
INDEX_PATH = KNOWLEDGE_DIR / "vector_index.json"


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split markdown by ## headings while keeping filename context."""
    chunks = []
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                body = "\n".join(current_lines).strip()
                chunks.append({"source": source, "text": f"[{source}]\n{body}"})
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        body = "\n".join(current_lines).strip()
        chunks.append({"source": source, "text": f"[{source}]\n{body}"})
    return [c for c in chunks if len(c["text"]) > 30]


def build_or_load_index(force_rebuild: bool = False) -> list[dict]:
    """Index the 11 Markdown policy files using Ollama embeddings."""
    if INDEX_PATH.exists() and not force_rebuild:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    index_data = []
    for md_file in md_files:
        if md_file.name.startswith("00_README"):
            continue
        chunks = _chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.name)
        for chunk in chunks:
            resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])
            chunk["embedding"] = resp["embedding"]
            index_data.append(chunk)

    INDEX_PATH.write_text(json.dumps(index_data), encoding="utf-8")
    return index_data


def query_agent_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve top-k relevant policy chunks for Dum-E."""
    index_data = build_or_load_index()
    q_emb = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"])

    scored = []
    for item in index_data:
        doc_emb = np.array(item["embedding"])
        sim = float(np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb) + 1e-9))
        scored.append({
            "source": item["source"],
            "similarity": round(sim, 4),
            "policy_text": item["text"],
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]