"""Text preprocessing shared by training and inference."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9']+")


def preprocess(text: str) -> str:
    """lowercase -> regex tokenise -> drop stop-words & 1-char tokens.
    Digits are kept on purpose: phone numbers / prices are strong spam signals."""
    toks = TOKEN_RE.findall(text.lower())
    return " ".join(t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 1)
