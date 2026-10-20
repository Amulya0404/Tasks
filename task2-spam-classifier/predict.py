"""Interactive spam prediction.  Run:  python predict.py
Or one-shot:  python predict.py "You won a free prize! Call now"
"""
import sys
from pathlib import Path
import joblib

# `preprocess` must be importable (module text_utils) for joblib to unpickle the pipeline
from text_utils import preprocess  # noqa: F401

PIPE = joblib.load(Path(__file__).parent / "spam_pipeline.joblib")


def predict_message(text: str) -> dict:
    """Return label + spam probability for one message."""
    p = float(PIPE.predict_proba([text])[0][1])
    return {"label": "SPAM" if p >= 0.5 else "HAM", "spam_probability": round(p, 4)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_message(" ".join(sys.argv[1:])))
    else:
        print("Spam classifier - type a message (empty line to quit)")
        try:
            while (msg := input("> ").strip()):
                r = predict_message(msg)
                print(f"  -> {r['label']}  (spam probability {r['spam_probability']:.2%})")
        except EOFError:
            pass
