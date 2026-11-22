"""RAGNova Studio — interface Streamlit.
Lancement (depuis la racine du projet) : streamlit run 05_Application/app.py"""
from pathlib import Path
import html, sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_core import Pipeline, load_results, METHODS, KEYS, OUT  # noqa: E402

st.set_page_config(page_title="RAGNova Studio", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root{--ink:#13233a;--muted:#5d6b80;--line:#e3e8ef;--teal:#0f7f6e;--teal-soft:#e7f4f1;--navy:#0f1e33;--amber:#b7791f;--amber-soft:#fbf3e4;--red:#b4413a;--red-soft:#fbecea;--bg:#f6f8fb}
html,body,[class*="css"]{font-family:"Segoe UI","Inter",system-ui,-apple-system,sans-serif;color:var(--ink)}
[data-testid="stAppViewContainer"]{background:var(--bg)}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1250px}
[data-testid="stSidebar"]{background:var(--navy)}
[data-testid="stSidebar"] *{color:#dfe7f1!important}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.45rem .6rem;border-radius:8px;margin-bottom:2px}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#ffffff12}
.brand{font-weight:800;font-size:1.45rem;letter-spacing:-.01em;margin:.2rem 0 .1rem}
.brand span{color:#4fd1b5!important}
.kicker{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--teal)}
.sb-kicker{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:#8ea3bd!important;font-weight:700}
.page-head{border-bottom:1px solid var(--line);padding-bottom:.9rem;margin-bottom:1.3rem}
.page-head h1{font-size:1.9rem;font-weight:750;margin:.2rem 0 .25rem;letter-spacing:-.02em}
.page-head p{color:var(--muted);margin:0;font-size:1rem}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:1rem 1.15rem;height:100%}
.card h4{margin:0 0 .35rem;font-size:1rem}
.card p{margin:0;color:var(--muted);font-size:.92rem}
.step{font-size:.75rem;font-weight:700;color:var(--teal)}
.note{border-left:3px solid var(--teal);background:var(--teal-soft);padding:.7rem 1rem;border-radius:0 8px 8px 0;color:#24504a;font-size:.92rem;margin:.6rem 0}
.warn{border-left:3px solid var(--amber);background:var(--amber-soft);padding:.7rem 1rem;border-radius:0 8px 8px 0;color:#6b4a12;font-size:.92rem;margin:.6rem 0}
.decision{display:flex;gap:1.2rem;align-items:center;background:#fff;border:1px solid var(--line);border-radius:12px;padding:1rem 1.2rem;margin:.4rem 0 1rem}
.badge{font-size:1.5rem;font-weight:800;padding:.35rem 1.1rem;border-radius:10px}
.badge.yes{background:var(--teal-soft);color:var(--teal)} .badge.no{background:var(--red-soft);color:var(--red)}
.decision .meta{color:var(--muted);font-size:.9rem;line-height:1.45}
.bar{height:8px;background:#edf1f6;border-radius:6px;overflow:hidden;width:220px;margin-top:.35rem}
.bar i{display:block;height:100%;background:var(--teal)}
.src{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.src.first{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal) inset}
.src-head{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin-bottom:.35rem}
.src-title{font-weight:700;font-size:1rem}
.rank{display:inline-block;min-width:1.9rem;font-weight:800;color:var(--teal)}
.score{font-family:Consolas,"SFMono-Regular",monospace;font-size:.82rem;background:#f1f4f8;border-radius:6px;padding:.15rem .5rem;color:var(--ink);white-space:nowrap}
.src-text{color:#36465c;font-size:.93rem;line-height:1.6}
.src-id{color:#8a97a8;font-size:.75rem;margin-top:.35rem}
mark{background:#fff1b8;padding:0 .1rem;border-radius:3px}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:.8rem 1rem;border-radius:12px}
.stButton button[kind="primary"]{background:var(--teal);border-color:var(--teal)}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = ["Do all mammals give birth to live young?", "is the great wall of china visible from space",
            "is a tomato a fruit or a vegetable", "can you get a tan through a window", "is the moon a planet"]


@st.cache_resource(show_spinner="Chargement des modèles sauvegardés…")
def pipeline():
    return Pipeline()


@st.cache_data
def results():
    return load_results()


@st.cache_data(show_spinner=False, max_entries=64)
def run(q):
    s = pipeline().all_scores(q)
    return {k: v for k, v in s.items()}


def head(kicker, title, sub):
    st.markdown(f'<div class="page-head"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def highlight(text, q, limit=None):
    import re
    from rag_core import tokens
    stop = {"is", "the", "a", "an", "of", "to", "in", "do", "does", "did", "are", "was", "were", "can", "and", "or", "on", "for", "it", "be", "has", "have", "all", "from", "you", "with", "at", "by", "as"}
    terms = sorted({t for t in tokens(q) if t not in stop and len(t) > 2}, key=len, reverse=True)
    t = text if not limit or len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"
    t = html.escape(t)
    if terms:
        t = re.sub(r"(?i)\b(" + "|".join(re.escape(html.escape(x)) for x in terms) + r")\b", r"<mark>\1</mark>", t)
    return t


def source_card(rank, doc, score, q, label="score", first=False, extra=""):
    st.markdown(
        f'<div class="src{" first" if first else ""}"><div class="src-head"><div class="src-title"><span class="rank">{rank:02d}</span>{html.escape(doc["title"])}</div>'
        f'<span class="score">{label} = {score:.4f}</span></div><div class="src-text">{highlight(doc["text"], q)}</div>'
        f'<div class="src-id">doc_id {doc["doc_id"]}{extra}</div></div>', unsafe_allow_html=True)


def question_input(key, default):
    c1, c2 = st.columns([3, 1])
    ex = c2.selectbox("Exemples", ["—"] + EXAMPLES, key=key + "_ex")
    q = c1.text_input("Question (en anglais, comme dans BoolQ)", value=ex if ex != "—" else default, key=key + "_q" + ex)
    return q


def empty_guard(q):
    if not q or not q.strip():
        st.markdown('<div class="warn"><b>Question vide.</b> Saisissez une question pour lancer la recherche.</div>', unsafe_allow_html=True)
        return True
    return False


def no_signal(s):
    if not s["_has_signal"]:
        st.markdown('<div class="warn"><b>Aucun terme de la question n’appartient au vocabulaire du corpus.</b> BM25 et TF-IDF valent 0 partout ; aucun passage ni décision n’est affiché, car ils ne seraient pas informatifs.</div>', unsafe_allow_html=True)
        return True
    return False


with st.sidebar:
    st.markdown('<div class="sb-kicker">École Centrale Casablanca</div><div class="brand">◆ RAG<span>Nova</span></div>', unsafe_allow_html=True)
    st.caption("Recherche documentaire sur BoolQ")
    page = st.radio("Navigation", ["Accueil", "Question & passages", "Comparer les méthodes", "Laboratoire reranking", "Évaluation", "Analyse d’erreurs", "Méthode & équipe"], label_visibility="collapsed")
    st.divider()
    st.caption("Modèles chargés depuis 04_Experiments/results/rag_nova_models.joblib · calcul local sur CPU.")

m, rows, unc = results()
P = pipeline()
NAMES = {"bm25": "BM25", "tfidf": "TF-IDF", "lsa": "LSA", "hybrid": "Fusion pondérée", "rrf": "RRF", "rerank": "Reranking"}


def metrics_df():
    return pd.DataFrame([{"Méthode": NAMES[k], **{c: m[k][c] for c in ("Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10")}} for k in NAMES])


if page == "Accueil":
    head("Prototype de recherche", "Interroger un corpus, inspecter les preuves", "Comparer recherche lexicale, sémantique, hybride et reranking sur le corpus BoolQ, passage par passage.")
    c = st.columns(4)
    c[0].metric("Passages indexés", f"{m['settings']['documents']:,}".replace(",", " "))
    c[1].metric("Questions de test", f"{m['settings']['test_queries']:,}".replace(",", " "))
    c[2].metric("Reranking · Hit@5", f"{m['rerank']['Hit@5']:.1%}")
    c[3].metric("Latence médiane (test)", f"{m['latency_e2e_ms']['p50']:.0f} ms")
    st.write("")
    steps = [("01", "Rechercher", "BM25, TF-IDF et LSA (SVD 48 dim.) notent les 12 259 passages."),
             ("02", "Combiner", f"Fusion pondérée (α BM25 = {m['settings']['fusion_alpha_bm25']:.1f}, choisi sur validation) et RRF (k = 60)."),
             ("03", "Réordonner", "Un reranker logistique note ~100 candidats à partir de 4 signaux."),
             ("04", "Décider", "Un lecteur TF-IDF + régression logistique classe oui / non à partir du passage n°1.")]
    for col, (n, t, d) in zip(st.columns(4), steps):
        col.markdown(f'<div class="card"><div class="step">{n}</div><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Le lecteur produit uniquement une <b>décision binaire oui / non</b> avec une probabilité. Ce n’est pas un LLM et il ne génère aucun texte : les passages affichés sont des extraits du corpus.</div>', unsafe_allow_html=True)
    st.subheader("Résultats enregistrés sur le test interne (n = 2 083)")
    st.dataframe(metrics_df().style.format({c: "{:.3f}" for c in ("Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10")}).highlight_max(subset=["Hit@5", "MRR@10", "nDCG@10"], color="#e7f4f1"), hide_index=True, width="stretch")

elif page == "Question & passages":
    head("Recherche interactive", "Question & passages", "Posez une question fermée ; le pipeline remonte les passages et le lecteur rend une décision oui / non.")
    q = question_input("qa", EXAMPLES[0])
    c1, c2 = st.columns([2, 1])
    method = c1.selectbox("Stratégie de recherche", METHODS, index=5)
    k = c2.slider("Nombre de passages", 1, 10, 5)
    if st.button("Rechercher", type="primary") or st.session_state.get("qa_done") == (q, method, k):
        st.session_state["qa_done"] = (q, method, k)
        if not empty_guard(q):
            s = run(q)
            if no_signal(s): st.stop()
            res = P.top(s, KEYS[method], k)
            dec, p = P.read(q, res[0][0]["text"])
            cls = "yes" if dec == "Oui" else "no"
            st.markdown(f'<div class="decision"><div class="badge {cls}">{dec}</div><div class="meta"><b>Décision du lecteur binaire</b> · P(oui) = {p:.1%}<div class="bar"><i style="width:{p*100:.1f}%"></i></div>'
                        f'Fondée sur le passage n°1 « {html.escape(res[0][0]["title"])} ». Classifieur TF-IDF + régression logistique, exactitude 56,8 % sur le test interne — ce n’est pas une réponse générée.</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Passages sources · {method}**")
            lab = "P(pertinent)" if method == "Reranking" else "score"
            for i, (d, sc, _) in enumerate(res, 1):
                source_card(i, d, sc, q, lab, first=i == 1)

elif page == "Comparer les méthodes":
    head("Inspection des méthodes", "Comparer les méthodes", "Les six stratégies appliquées à la même question, côte à côte, avec leurs scores bruts.")
    q = question_input("cmp", EXAMPLES[1]); k = st.slider("Top-k", 3, 10, 5)
    if st.button("Comparer", type="primary") or st.session_state.get("cmp_done") == (q, k):
        st.session_state["cmp_done"] = (q, k)
        if not empty_guard(q):
            s = run(q)
            if no_signal(s): st.stop()
            table = {}
            for meth in METHODS:
                table[meth] = [f"{d['title']}  ({sc:.3f})" for d, sc, _ in P.top(s, KEYS[meth], k)]
            st.dataframe(pd.DataFrame(table, index=[f"#{i}" for i in range(1, k + 1)]), width="stretch")
            st.caption("Score entre parenthèses : BM25 brut, cosinus TF-IDF / LSA, score normalisé de fusion, somme RRF 1/(60+rang), probabilité du reranker.")
            meth = st.radio("Détail des passages pour", METHODS, horizontal=True)
            for i, (d, sc, _) in enumerate(P.top(s, KEYS[meth], k), 1):
                source_card(i, d, sc, q, first=i == 1)

elif page == "Laboratoire reranking":
    head("Reranking", "Laboratoire reranking", "Comment le reranker logistique réordonne les candidats issus de BM25, TF-IDF, LSA et RRF.")
    q = question_input("rr", EXAMPLES[0]); k = st.slider("Candidats affichés", 3, 15, 8)
    if st.button("Analyser", type="primary") or st.session_state.get("rr_done") == (q, k):
        st.session_state["rr_done"] = (q, k)
        if not empty_guard(q):
            s = run(q)
            if no_signal(s): st.stop()
            cand, F = s["_cand"], s["_feats"]
            coef = P.a["reranker"].coef_[0]; inter = P.a["reranker"].intercept_[0]
            rb = {int(j): r for r, j in enumerate(np.argsort(-s["bm25"]), 1) if r <= 200}
            rrf_r = {int(j): r for r, j in enumerate(np.argsort(-s["rrf"]), 1) if r <= 200}
            order = np.argsort(-s["rerank"][cand])[:k]
            df = pd.DataFrame([{"Rang rerank": i + 1, "Rang BM25": str(rb.get(int(cand[o]), ">200")), "Rang RRF": str(rrf_r.get(int(cand[o]), ">200")),
                                "Titre": P.docs[int(cand[o])]["title"], "BM25 norm.": F[o, 0], "TF-IDF norm.": F[o, 1], "Cos. LSA": F[o, 2],
                                "Recouvr. titre": F[o, 3], "P(pertinent)": s["rerank"][cand[o]]} for i, o in enumerate(order)])
            c = st.columns(3)
            for col, (lab, val) in zip(c, [("Candidats notés par le reranker", str(len(cand))), ("Passage n°1 avec BM25", P.docs[int(np.argmax(s["bm25"]))]["title"]), ("Passage n°1 après reranking", P.docs[int(cand[order[0]])]["title"])]):
                col.markdown(f'<div class="card"><p>{lab}</p><h4 style="margin-top:.3rem;font-size:1.15rem">{html.escape(val)}</h4></div>', unsafe_allow_html=True)
            st.write("")
            st.dataframe(df.style.format({x: "{:.3f}" for x in ("BM25 norm.", "TF-IDF norm.", "Cos. LSA", "Recouvr. titre", "P(pertinent)")}), hide_index=True, width="stretch")
            st.markdown("**Poids appris du reranker** (régression logistique équilibrée, 1 600 questions d’entraînement)")
            st.dataframe(pd.DataFrame([{"Signal": n, "Coefficient": v} for n, v in zip(["BM25 normalisé", "TF-IDF normalisé", "Cosinus LSA", "Recouvrement question–titre"], coef)] + [{"Signal": "Constante", "Coefficient": inter}]).style.format({"Coefficient": "{:+.3f}"}), hide_index=True)
            st.markdown("**Passages après reranking**")
            for i, o in enumerate(order[:3], 1):
                source_card(i, P.docs[int(cand[o])], float(s["rerank"][cand[o]]), q, "P(pertinent)", first=i == 1, extra=f" · rang BM25 {rb.get(int(cand[o]), '>200')}")

elif page == "Évaluation":
    head("Résultats expérimentaux", "Évaluation", "Mesures enregistrées par 04_Experiments sur le test interne (titres disjoints), paramètres choisis sur validation.")
    st.markdown('<div class="note">Valeurs lues dans <code>metrics.json</code> et <code>uncertainty.json</code> ; rien n’est recalculé ni modifié ici. Le split Kaggle public n’est pas étiqueté : l’évaluation porte sur une partition interne étiquetée (n = 2 083).</div>', unsafe_allow_html=True)
    df = metrics_df()
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns[1:]}).highlight_max(subset=list(df.columns[1:]), color="#e7f4f1"), hide_index=True, width="stretch")
    metric = st.radio("Métrique", ["Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10"], index=1, horizontal=True)
    st.bar_chart(df.set_index("Méthode")[[metric]], color="#0f7f6e", horizontal=True, height=280)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lecteur binaire oui / non** (contexte = passage n°1 du reranking)")
        r = m["reader_top_reranked_context"]
        a, b = st.columns(2); a.metric("Exactitude", f"{r['accuracy']:.1%}"); b.metric("F1 (oui)", f"{r['f1_yes']:.1%}")
        a.metric("Précision (oui)", f"{r['precision_yes']:.1%}"); b.metric("Rappel (oui)", f"{r['recall_yes']:.1%}")
    with c2:
        st.markdown("**Latence de bout en bout** (CPU, représentations déjà chargées)")
        L = m["latency_e2e_ms"]; a, b = st.columns(2); a.metric("Médiane", f"{L['p50']:.1f} ms"); b.metric("p95", f"{L['p95']:.1f} ms")
        st.markdown("**Choix de α sur validation (Hit@5)**")
        tun = pd.read_csv(OUT / "validation_fusion_tuning.csv"); tun["alpha_bm25"] = tun["alpha_bm25"].round(1)
        st.line_chart(tun.set_index("alpha_bm25"), color="#0f7f6e", height=170)
    if unc:
        st.markdown("**Incertitude · comparaisons appariées (Hit@5)**")
        st.dataframe(pd.DataFrame([{"Comparaison": k.replace("_hit5", "").replace("_vs_", " vs ").replace("rerank", "Reranking").replace("rrf", "RRF").replace("bm25", "BM25"),
                                    "Différence": v["difference"], "IC 95 % bas": v["bootstrap_95ci"][0], "IC 95 % haut": v["bootstrap_95ci"][1], "p (permutation)": v["paired_cluster_sign_flip_p_approx"], "Titres": v["cluster_count"]} for k, v in unc.items()]).style.format({"Différence": "{:+.3f}", "IC 95 % bas": "{:+.3f}", "IC 95 % haut": "{:+.3f}", "p (permutation)": "{:.4f}"}), hide_index=True, width="stretch")
        st.caption("Bootstrap par titres (5 000 réplications) et permutation signée appariée au niveau titre.")
    st.markdown(f'<div class="warn">α retenu = {m["settings"]["fusion_alpha_bm25"]:.1f} : la fusion pondérée se réduit donc à BM25 seul, d’où des scores identiques à BM25.</div>', unsafe_allow_html=True)

elif page == "Analyse d’erreurs":
    head("Analyse d’erreurs", "Où le pipeline se trompe", "Traces enregistrées du test interne : erreurs du lecteur et échecs de récupération.")
    err = [x for x in rows if x["answer"] != x["prediction"]]; miss = [x for x in rows if x["retrieved_doc_id"] != x["gold_doc_id"]]
    c = st.columns(3); c[0].metric("Questions de test", len(rows)); c[1].metric("Erreurs du lecteur", len(err), f"{len(err)/len(rows):.1%}", delta_color="off"); c[2].metric("Passage n°1 ≠ passage attendu", len(miss), f"{len(miss)/len(rows):.1%}", delta_color="off")
    a, b = st.columns([2, 1]); term = a.text_input("Filtrer par mots de la question", ""); kind = b.selectbox("Type", ["Erreurs du lecteur", "Échecs de récupération"])
    src = err if kind == "Erreurs du lecteur" else miss
    sel = [x for x in src if term.casefold() in x["question"].casefold()]
    st.caption(f"{len(sel)} cas · 40 premiers affichés")
    for x in sel[:40]:
        with st.expander(x["question"]):
            st.write(f"Attendu : **{'Oui' if x['answer'] else 'Non'}** · prédit : **{'Oui' if x['prediction'] else 'Non'}** · P(oui) = {x['answer_probability_yes']:.3f}")
            st.write(f"Passage n°1 : **{x['retrieved_title']}** {'✓ passage attendu' if x['retrieved_doc_id']==x['gold_doc_id'] else '✗ différent du passage attendu'}")
            st.markdown(f'<div class="src-text">{highlight(x["top_context"], x["question"])}</div>', unsafe_allow_html=True)

else:
    head("Notes de recherche", "Méthode & équipe", "RAGNova Studio · École Centrale Casablanca")
    st.markdown("**Zakaria Midine · Yassine Ouassour · Adam Chafay · Mehdi Elmalki**")
    st.markdown(f"""
- **Données** : BoolQ (Kaggle, version 2, licence CC BY-SA 3.0). Titres répartis de façon déterministe train / validation / test (70/15/15) pour limiter les fuites.
- **Recherche** : BM25 (k1 = 1,5, b = 0,75), TF-IDF (uni+bigrammes), LSA (SVD {m['settings']['svd_components']} dimensions).
- **Fusion** : combinaison convexe BM25 / TF-IDF normalisés (α choisi sur validation) et Reciprocal Rank Fusion (k = 60, listes top-100).
- **Reranking** : régression logistique équilibrée sur 4 signaux, candidats = union des top-30 RRF, BM25, TF-IDF et LSA.
- **Lecteur** : TF-IDF + régression logistique, décision binaire oui / non. **Aucun modèle génératif.**
- MiniLM, cross-encoder et FLAN-T5 sont des pistes d’extension : non exécutés, non mesurés.
- Graine : {m['settings']['seed']}. Protocole détaillé dans `06_Documentation`.
""")
