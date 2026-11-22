"""Pipeline RAGNova : reproduit exactement la logique de 04_Experiments/run_experiments.py
à partir des artefacts sauvegardés (rag_nova_models.joblib)."""
from pathlib import Path
import json, re, warnings
import numpy as np
import joblib
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04_Experiments" / "results"
METHODS = ["BM25", "TF-IDF", "LSA", "Fusion pondérée", "RRF", "Reranking"]
KEYS = {"BM25": "bm25", "TF-IDF": "tfidf", "LSA": "lsa", "Fusion pondérée": "hybrid", "RRF": "rrf", "Reranking": "rerank"}


def tokens(s):
    return re.findall(r"(?u)\b\w+\b", s.casefold())


def rankings(scores, k=10):
    k = min(k, len(scores))
    p = np.argpartition(-scores, k - 1)[:k]
    return p[np.argsort(-scores[p])]


class Pipeline:
    def __init__(self, path=OUT / "rag_nova_models.joblib"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # échoue si la version scikit-learn ne correspond pas
            a = joblib.load(path)
        self.a = a
        self.docs = a["docs"]
        self.C = a["count_matrix"].tocsc()
        self.n = len(self.docs)

    def bm25(self, q):
        a = self.a; qv = a["count"].transform([q]); vals = np.zeros(self.n, np.float32)
        for t in qv.indices:
            s, e = self.C.indptr[t], self.C.indptr[t + 1]
            rows = self.C.indices[s:e]; tf = self.C.data[s:e]
            den = tf + 1.5 * (1 - 0.75 + 0.75 * a["doc_lengths"][rows] / a["avgdl"])
            vals[rows] += a["bm25_idf"][t] * (tf * (1.5 + 1)) / (den + 1e-9)
        return vals

    def tfidf(self, q):
        return (self.a["doc_tfidf"] @ self.a["tfidf"].transform([q]).T).toarray().ravel()

    def lsa(self, q):
        a = self.a
        z = normalize(a["svd"].transform(a["tfidf"].transform([q])).astype(np.float32))
        return (a["doc_lsa"] @ z.T).ravel()

    def all_scores(self, q):
        """Retourne les scores des 6 méthodes pour tout le corpus."""
        q = (q or "").strip()
        if not q:
            raise ValueError("La question est vide.")
        b, t, l = self.bm25(q), self.tfidf(q), self.lsa(q)
        al = self.a["fusion_alpha"]
        hybrid = al * b / (b.max() + 1e-9) + (1 - al) * t / (t.max() + 1e-9)
        rrf = np.zeros(self.n, np.float32)
        for arr in (b, t, l):
            for rank, j in enumerate(rankings(arr, 100), 1):
                rrf[j] += 1 / (60 + rank)
        cand = np.unique(np.r_[rankings(rrf, 30), rankings(b, 30), rankings(t, 30), rankings(l, 30)])
        qt = set(tokens(q)); feats = []
        for j in cand:
            ov = len(qt & set(tokens(self.docs[j]["title"]))) / max(1, len(qt))
            feats.append([b[j] / (b.max() + 1e-9), t[j] / (t.max() + 1e-9), l[j], ov])
        feats = np.asarray(feats)
        rs = self.a["reranker"].predict_proba(feats)[:, 1]
        rer = np.full(self.n, -1.0, np.float32); rer[cand] = rs
        return {"bm25": b, "tfidf": t, "lsa": l, "hybrid": hybrid, "rrf": rrf, "rerank": rer,
                "_cand": cand, "_feats": feats, "_has_signal": bool(b.max() > 0 or t.max() > 0)}

    def top(self, scores, key, k=5):
        return [(self.docs[int(i)], float(scores[key][i]), int(i)) for i in rankings(scores[key], k)]

    def read(self, q, passage):
        p = float(self.a["reader"].predict_proba([q + " [SEP] " + passage])[0, 1])
        return ("Oui" if p >= 0.5 else "Non"), p


def load_results():
    m = json.loads((OUT / "metrics.json").read_text(encoding="utf8"))
    rows = [json.loads(x) for x in (OUT / "per_question_traces.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    u = json.loads((OUT / "uncertainty.json").read_text(encoding="utf8")) if (OUT / "uncertainty.json").exists() else {}
    return m, rows, u
