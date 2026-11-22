from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(__file__).resolve().parent / "raw" / "boolq_dataset_v2.zip"
OUT = ROOT / "04_Experiments" / "data"
SEED = 20261114


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()


def split_for_title(title: str) -> str:
    v = int(hashlib.sha256(f"{SEED}|{norm(title)}".encode()).hexdigest()[:8], 16) % 100
    return "train" if v < 70 else "validation" if v < 85 else "test"


def main() -> None:
    if not ZIP.exists():
        raise FileNotFoundError(f"Placez l'archive Kaggle à {ZIP}")
    OUT.mkdir(parents=True, exist_ok=True)
    labeled, official_test = [], []
    with zipfile.ZipFile(ZIP) as z:
        for name, target in (("train.jsonl", labeled), ("dev.jsonl", labeled), ("test.jsonl", official_test)):
            with z.open(name) as f:
                for raw in f:
                    row = json.loads(raw)
                    row["source_split"] = name.removesuffix(".jsonl")
                    target.append(row)

    documents: dict[str, dict] = {}
    queries = []
    for i, row in enumerate(labeled):
        passage = re.sub(r"\s+", " ", row["passage"]).strip()
        title = re.sub(r"\s+", " ", row["title"]).strip()
        doc_id = hashlib.sha256(norm(passage).encode()).hexdigest()[:20]
        doc = documents.setdefault(doc_id, {"doc_id": doc_id, "title": title, "text": passage, "source_splits": []})
        if row["source_split"] not in doc["source_splits"]:
            doc["source_splits"].append(row["source_split"])
        queries.append({
            "qid": f"bq_{i:05d}", "question": re.sub(r"\s+", " ", row["question"]).strip(),
            "answer": bool(row["answer"]), "gold_doc_id": doc_id, "title": title,
            "group_split": split_for_title(title), "source_split": row["source_split"],
        })

    # The public unlabeled split contributes only documents to the searchable corpus.
    for row in official_test:
        passage = re.sub(r"\s+", " ", row["passage"]).strip()
        title = re.sub(r"\s+", " ", row["title"]).strip()
        doc_id = hashlib.sha256(norm(passage).encode()).hexdigest()[:20]
        doc = documents.setdefault(doc_id, {"doc_id": doc_id, "title": title, "text": passage, "source_splits": []})
        if "official_unlabeled_test" not in doc["source_splits"]:
            doc["source_splits"].append("official_unlabeled_test")

    with (OUT / "queries_labeled.jsonl").open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "documents.jsonl").open("w", encoding="utf-8") as f:
        for row in documents.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "queries_official_test_unlabeled.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(official_test):
            f.write(json.dumps({"qid": f"official_test_{i:05d}", "question": row["question"], "title": row["title"], "gold_doc_id": hashlib.sha256(norm(row["passage"]).encode()).hexdigest()[:20]}, ensure_ascii=False) + "\n")

    counts = {s: sum(q["group_split"] == s for q in queries) for s in ("train", "validation", "test")}
    summary = {
        "dataset": "BoolQ / Kaggle averkij/boolq-dataset", "version": 2, "license": "CC BY-SA 3.0",
        "seed": SEED, "labeled_queries": len(queries), "official_unlabeled_queries": len(official_test),
        "unique_passages": len(documents), "query_counts_by_group_title": counts,
        "unique_titles": len({norm(q["title"]) for q in queries}),
        "unique_questions": len({norm(q["question"]) for q in queries}),
        "answer_balance": {"yes": sum(q["answer"] for q in queries), "no": sum(not q["answer"] for q in queries)},
        "split_definition": "70/15/15 deterministic SHA-256 assignment by normalized Wikipedia title; one title cannot cross query splits",
        "corpus_policy": "All public passages, including official unlabeled-test passages, form the retrieval corpus; only labeled train/validation/test questions are evaluated.",
        "leakage_policy": "Answers and query labels do not enter the index. Model fitting uses train titles only; tuning uses validation titles only; test labels are scored once after freezing settings.",
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
