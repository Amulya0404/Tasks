# Task 4 — Document Question-Answering Assistant

CodeNova AI/ML Internship · Batch 01

A fully local, extractive Q&A assistant. It answers **only** with text found in the provided documents, always shows the supporting passage, and says **"Information not found in provided document."** when it cannot ground an answer. No paid APIs, no model downloads, no generation → nothing to hallucinate.

## How it works
1. **Ingest & chunk** — every `.txt` file in `corpus/` is split into paragraph-level chunks (very short paragraphs merged).
2. **Retrieve** — TF-IDF (stemmed unigrams + bigrams) with cosine similarity returns the top passages.
3. **Extract** — the best-matching sentence(s) of the top passage form the answer.
4. **Grounding gate** — an answer is given only if (a) cosine similarity ≥ 0.12 **and** (b) ≥ 60% of the question's content words appear in the passage. Otherwise the fallback message is returned.

## Run
```bash
pip install -r ../requirements.txt
python qa_assistant.py                         # interactive
python qa_assistant.py "What is a model card?" # one-shot
python qa_assistant.py --demo                  # 8 answerable + 6 unanswerable questions
```
Corpus: `corpus/` (3 original sample documents on ML basics, NLP/retrieval and responsible AI, 22 chunks). Drop in any `.txt` files to use your own. Full transcript: `demo_transcript.txt`.

## Demo results
**7 / 8** answerable questions answered with citations · **6 / 6** unanswerable questions correctly refused · 0 hallucinated answers.

### Grounded answer
```
Q: What is a model card?
A: Model cards document the intended use, training data, evaluation results and known limitations of a model.
   Source: responsible_ai.txt (chunk 20) | similarity=0.27 coverage=1.0
   Supporting passage: "Privacy requires that personal data is collected only when necessary ... Model cards document the intended use, training data, evaluation results and known limitations of a model. ..."
```
### Another
```
Q: Why should the test set not be used during tuning?
A: A dataset is usually divided into a training set and a test set. The test set must never be used while training or tuning, because doing so causes data leakage and gives an unrealistically optimistic estimate of performance.
   Source: machine_learning_basics.txt (chunk 4) | similarity=0.514 coverage=1.0
```
### Unsupported questions
```
Q: Who won the 2018 FIFA World Cup?
A: Information not found in provided document.

Q: Explain how quantum computers factor large numbers.
A: Information not found in provided document.
```
(See `demo_transcript.txt` for every question with its full cited passage.)

## Known limitations
- Retrieval is lexical: a paraphrase with no shared vocabulary is missed. "How is bias introduced into a system?" is refused because the document says bias *enters*; the gate fails safe (refuses) instead of guessing.
- Answers are extracted sentences, not synthesised — an answer can include a neighbouring sentence that is only loosely relevant.
- The thresholds were set on this small corpus and should be re-tuned for a different one. For semantic matching, swap TF-IDF for free local `sentence-transformers` embeddings.
