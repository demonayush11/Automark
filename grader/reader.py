"""Read a whole answer script (photos / PDF pages) into text with our own model, and check
whether each page was readable enough to trust.

There is one reader: our CNN + BiLSTM + CTC model (htr_model.py) with dictionary correction
(lexicon.py). When a page is too hard to read reliably, the teacher is asked to re-upload a
clearer photo (page_quality). Everything runs on this computer.
"""
import re

import cv2
import numpy as np

from .htr_model import HandwritingModel
from .segment import load_pages, segment_page, draw_boxes

# ---------------------------------------------------------------- thresholds
LOW_CONF = 0.5            # a word whose confidence is below this is "unsure" and gets [?]
WARN_UNSURE = 0.15        # unsure share at or above this -> "warn"
BAD_UNSURE = 0.25         # unsure share at or above this -> "bad"
TILT_WARN_DEG = 14.0      # tilt correction searches up to +-15 deg; at or beyond 14 the page may be tilted more
BLUR_WARN_VAR = 18.0      # variance of the Laplacian of the cleaned page below this -> blurry
                          # (calibrated with evaluate.py: blur_3 / blur+noise photos score 5-12,
                          #  every readable case scores 25 or more)
FEW_WORDS_PER_LINE = 3    # median words per line below this ...
FEW_WORDS_MIN_LINES = 5   # ... on a page with at least this many lines -> words were probably missed

LEVELS = ("ok", "warn", "bad")

RETAKE_TIPS = ["Lay the page flat.",
               "Use even light with no shadow on the page.",
               "Hold the camera straight above the page.",
               "Fill the frame with the page.",
               "Hold the phone steady (no motion blur).",
               "Or use a scanner / scanning app instead of a photo."]

_model = None


def get_model():
    global _model
    if _model is None:
        _model = HandwritingModel()
    return _model


def _fix_spacing(text):
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


def _own_read_page(model, bgr):
    gray, lines, info = segment_page(bgr)
    flat = [w["img"] for line in lines for w in line]
    results = model.read_words(flat) if flat else []
    k, page_lines, text_lines = 0, [], []
    words = low = empty = 0
    for line in lines:
        row = results[k:k + len(line)]
        k += len(line)
        page_lines.append(row)
        parts = []
        for word, conf in row:
            if not word:              # the model found ink but could not read anything
                empty += 1
                parts.append("[?]")
                continue
            words += 1
            if conf < LOW_CONF:
                low += 1
                parts.append(word + "[?]")
            else:
                parts.append(word)
        text_lines.append(_fix_spacing(" ".join(parts)))
    return {"gray": gray, "lines": lines, "results": results, "page_lines": page_lines,
            "text": "\n".join(text_lines), "words": words, "low": low, "empty": empty,
            "angle": info["angle"]}


def blur_score(gray):
    """Sharpness: variance of the Laplacian (edge strength). Low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def page_quality(own):
    """Decide whether a page was read reliably enough.
    Returns {"level": "ok"|"warn"|"bad", "reasons": [...], "unsure_pct": float, "words": int}."""
    readings = own["words"] + own["empty"]
    unsure = own["low"] + own["empty"]
    share = unsure / readings if readings else 1.0
    level, reasons = "ok", []

    def flag(lvl, reason):
        nonlocal level
        if LEVELS.index(lvl) > LEVELS.index(level):
            level = lvl
        reasons.append(reason)

    if not own["lines"] or readings == 0:
        flag("bad", "No handwriting found on the page.")
        return {"level": level, "reasons": reasons, "unsure_pct": 100.0, "words": 0}
    if share >= BAD_UNSURE:
        flag("bad", f"The model could not read {round(100 * share)}% of the words.")
    elif share >= WARN_UNSURE:
        flag("warn", f"The model was unsure about {round(100 * share)}% of the words.")
    n_lines = len(own["lines"])
    per_line = float(np.median([len(line) for line in own["lines"]]))
    if n_lines >= FEW_WORDS_MIN_LINES and per_line < FEW_WORDS_PER_LINE:
        flag("warn", f"Very few words per line (about {per_line:g}); some words may have been missed.")
    if abs(own["angle"]) >= TILT_WARN_DEG:
        flag("warn", f"The page is strongly tilted (about {abs(own['angle']):.0f}° or more).")
    if blur_score(own["gray"]) < BLUR_WARN_VAR:
        flag("warn", "The photo looks blurry.")
    return {"level": level, "reasons": reasons, "unsure_pct": round(100 * share, 1), "words": readings}


def read_script(paths, preview_prefix=None):
    """paths: one student's files in page order.
    Returns dict(text, pages, words, low_conf, quality, reasons)."""
    model = get_model()
    pages_out, texts = [], []
    words_total = low_total = 0
    level, reasons = "ok", []
    page_no = 0
    for path in paths:
        for bgr in load_pages(path):
            page_no += 1
            own = _own_read_page(model, bgr)
            q = page_quality(own)
            words_total += own["words"] + own["empty"]
            low_total += own["low"] + own["empty"]
            texts.append(own["text"])
            if LEVELS.index(q["level"]) > LEVELS.index(level):
                level = q["level"]
            reasons += [f"Page {page_no}: {r}" for r in q["reasons"]]
            preview = None
            if preview_prefix:
                preview = f"{preview_prefix}_p{page_no}.jpg"
                cv2.imwrite(preview, draw_boxes(own["gray"], own["lines"], own["results"]),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            pages_out.append({"lines": own["page_lines"], "preview": preview, "quality": q})
    if page_no == 1:                         # one page: no need to say "Page 1"
        reasons = [r.split(": ", 1)[1] for r in reasons]
    return {"text": "\n\n".join(texts).strip(), "pages": pages_out, "words": words_total,
            "low_conf": low_total, "quality": level, "reasons": reasons}
