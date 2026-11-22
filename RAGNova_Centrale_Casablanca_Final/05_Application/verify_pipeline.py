"""Vérifie que l'application reproduit les traces enregistrées du test interne."""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from rag_core import Pipeline, load_results, rankings
requested = int(sys.argv[1]) if len(sys.argv) > 1 else 300
if requested < 1:
    raise SystemExit("Le nombre de questions doit être positif.")
p = Pipeline(); m, rows, _ = load_results(); ids = [d["doc_id"] for d in p.docs]
N = min(requested, len(rows))
ok = {k: 0 for k in ("bm25", "tfidf", "lsa", "hybrid", "rrf", "rerank")}; rd = 0; t0 = time.time()
for x in rows[:N]:
    s = p.all_scores(x["question"])
    for k in ok:
        ok[k] += [ids[i] for i in rankings(s[k], 10)] == x["rankings"][k]
    top = int(rankings(s["rerank"], 1)[0]); d, pr = p.read(x["question"], p.docs[top]["text"])
    rd += (d == "Oui") == x["prediction"] and abs(pr - x["answer_probability_yes"]) < 1e-6
print(f"{N} questions, {time.time()-t0:.1f}s")
for k, v in ok.items(): print(f"  top-10 identique {k:7s}: {v}/{N}")
print(f"  décision lecteur + P(oui) identiques: {rd}/{N}")
if rd != N or any(v != N for v in ok.values()):
    raise SystemExit(1)
