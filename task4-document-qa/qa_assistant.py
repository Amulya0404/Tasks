"""Task 4 - Document Question-Answering Assistant (CodeNova AI/ML Batch 01).

100% local, no paid APIs. Pipeline:
  1. Ingest every .txt file in ./corpus and split into paragraph chunks (semantic units).
  2. Index chunks with TF-IDF (unigrams+bigrams) and search with cosine similarity.
  3. Extractive answer: pick the best-matching sentence(s) of the top chunk.
  4. Grounding gate: answer ONLY if the retrieval score AND the share of the question's
     content words found in the passage both clear a threshold; otherwise reply
     "Information not found in provided document."  (no generation => no hallucination)

Usage:
    python qa_assistant.py                      # interactive
    python qa_assistant.py "What is recall?"    # one-shot
    python qa_assistant.py --demo               # run the built-in demo / test questions
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = Path(__file__).parent / "corpus"
NOT_FOUND = "Information not found in provided document."
MIN_SCORE = 0.12      # min cosine similarity of best chunk
MIN_COVERAGE = 0.60   # min share of question content-words present in best chunk
MIN_CHUNK_WORDS = 15
QUESTION_WORDS = {"what", "which", "who", "whom", "when", "where", "why", "how", "does", "do",
                  "is", "are", "can", "explain", "describe", "tell", "define", "mean", "means",
                  "used", "use", "work", "works", "matter", "matters", "differ", "differs", "difference"}


def stem(w: str) -> str:
    """Very light suffix stripper so 'classifiers' ~ 'classifier', 'memorises' ~ 'memoris'."""
    for suf in ("ings", "ing", "ies", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def content_terms(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {stem(t) for t in toks if t not in ENGLISH_STOP_WORDS and t not in QUESTION_WORDS}


def analyzer(text: str):
    toks = [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in ENGLISH_STOP_WORDS]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


@dataclass
class Chunk:
    source: str
    idx: int
    heading: str
    text: str


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list:
    chunks = []
    for f in sorted(corpus_dir.glob("*.txt")):
        paras = [p.strip() for p in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")) if p.strip()]
        heading, body = paras[0], paras[1:]
        buf = ""
        for p in body:  # one paragraph = one chunk; only very short paragraphs are merged
            if buf:
                chunks.append(Chunk(f.name, len(chunks), heading, buf)); buf = ""
            if len(p.split()) < MIN_CHUNK_WORDS:
                buf = p
            else:
                chunks.append(Chunk(f.name, len(chunks), heading, p))
        if buf:
            chunks.append(Chunk(f.name, len(chunks), heading, buf))
    for i, c in enumerate(chunks):
        c.idx = i
    return chunks


def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class DocumentQA:
    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.chunks = load_chunks(corpus_dir)
        self.vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{c.heading}. {c.text}" for c in self.chunks])

    def retrieve(self, question: str, k: int = 3):
        q = self.vec.transform([question])
        sims = cosine_similarity(q, self.matrix).ravel()
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        best, score = hits[0]
        q_terms = content_terms(question)
        coverage = (len(q_terms & content_terms(best.text + " " + best.heading)) / len(q_terms)
                    if q_terms else 0.0)
        out = {"question": question, "score": round(score, 3), "coverage": round(coverage, 2),
               "top_passages": [(c.source, c.idx, round(s, 3)) for c, s in hits]}
        if score < MIN_SCORE or coverage < MIN_COVERAGE:
            return {**out, "answered": False, "answer": NOT_FOUND}
        # extractive answer: sentences of the best chunk ranked by similarity to the question
        sents = split_sentences(best.text)
        s_sims = cosine_similarity(self.vec.transform([question]), self.vec.transform(sents)).ravel()
        order = np.argsort(-s_sims)
        chosen = [order[0]] + [i for i in order[1:2] if s_sims[i] >= 0.6 * s_sims[order[0]] and s_sims[i] > 0]
        answer = " ".join(sents[i] for i in sorted(chosen))
        return {**out, "answered": True, "answer": answer,
                "source": best.source, "chunk": best.idx, "passage": best.text}


def format_result(r: dict) -> str:
    lines = [f"Q: {r['question']}"]
    if r["answered"]:
        lines += [f"A: {r['answer']}",
                  f"   Source: {r['source']} (chunk {r['chunk']}) | similarity={r['score']} coverage={r['coverage']}",
                  f"   Supporting passage: \"{r['passage']}\""]
    else:
        lines += [f"A: {r['answer']}",
                  f"   (best similarity={r['score']}, coverage={r['coverage']} - below grounding threshold)"]
    return "\n".join(lines)


DEMO_SUPPORTED = [
    "What is overfitting?",
    "What does precision mean?",
    "How does BM25 differ from TF-IDF?",
    "Why should the test set not be used during tuning?",
    "What is retrieval-augmented generation?",
    "Why does chunking matter?",
    "What is a model card?",
    "How is bias introduced into a system?",
]
DEMO_UNSUPPORTED = [
    "Who won the 2018 FIFA World Cup?",
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "What is the price of Bitcoin?",
    "How many layers does ResNet-50 have?",
    "Explain how quantum computers factor large numbers.",
]

if __name__ == "__main__":
    qa = DocumentQA()
    args = sys.argv[1:]
    if args == ["--demo"]:
        ok = ref = 0
        for q in DEMO_SUPPORTED + DEMO_UNSUPPORTED:
            r = qa.answer(q); print(format_result(r), end="\n\n")
            if q in DEMO_SUPPORTED: ok += r["answered"]
            else: ref += not r["answered"]
        print(f"SUMMARY: answered {ok}/{len(DEMO_SUPPORTED)} answerable questions; "
              f"correctly refused {ref}/{len(DEMO_UNSUPPORTED)} unanswerable questions "
              f"({len(DEMO_UNSUPPORTED)-ref} hallucinated answers).")
    elif args:
        print(format_result(qa.answer(" ".join(args))))
    else:
        print(f"Loaded {len(qa.chunks)} chunks from {CORPUS_DIR.name}/. Ask a question (empty line to quit).")
        while (q := input("? ").strip()):
            print(format_result(qa.answer(q)), "\n")
