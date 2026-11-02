"""Read a whole answer script (photos / PDF pages) into text.

Reader modes:
  own   - only the project's own handwriting model
  auto  - own model first; if it is unsure about too many words on a page, that page is read by AI
  ai    - every page read by the AI vision model (Groq or Gemini)
"""
import re
import cv2
from .htr_model import HandwritingModel
from .segment import load_pages, segment_page, draw_boxes

_model = None
LOW_CONF = 0.5        # words below this are marked [?] for the teacher to check
AUTO_AI_SHARE = 0.25  # auto mode: if more than 25% of a page's words are unsure, read that page with AI


def get_model():
    global _model
    if _model is None:
        _model = HandwritingModel()
    return _model


def _fix_spacing(text):
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


def _own_read_page(model, bgr):
    gray, lines = segment_page(bgr)
    flat = [w["img"] for line in lines for w in line]
    results = model.read_words(flat) if flat else []
    k, page_lines, text_lines = 0, [], []
    words = low = 0
    for line in lines:
        row = results[k:k + len(line)]
        k += len(line)
        page_lines.append(row)
        parts = []
        for word, conf in row:
            if not word:
                continue
            words += 1
            if conf < LOW_CONF:
                low += 1
                parts.append(word + "[?]")
            else:
                parts.append(word)
        text_lines.append(_fix_spacing(" ".join(parts)))
    return {"gray": gray, "lines": lines, "results": results, "page_lines": page_lines,
            "text": "\n".join(text_lines), "words": words, "low": low}


def _looks_bad(r):
    """Own-model result too poor to trust: nothing found, or many unsure words."""
    if r["words"] == 0:
        return True
    return r["low"] / r["words"] > AUTO_AI_SHARE


def read_script(paths, preview_prefix=None, mode="own", settings=None):
    """paths: one student's files in page order.
    Returns dict(text, pages, words, low_conf, reader) where reader says who read it."""
    from . import ai_reader
    settings = settings or {}
    ai_engine = ai_reader.pick_engine(settings) if mode in ("auto", "ai") else None
    if mode == "ai" and not ai_engine:
        raise RuntimeError("Reading with AI needs a Groq or Gemini key (Settings page).")
    model = get_model()
    pages_out, texts = [], []
    words_total = low_total = 0
    used = set()
    ai_problems = []
    page_no = 0
    for path in paths:
        for bgr in load_pages(path):
            page_no += 1
            own = _own_read_page(model, bgr)
            text, w, lo, who = own["text"], own["words"], own["low"], "own"
            if ai_engine and (mode == "ai" or _looks_bad(own)):
                try:
                    text = ai_reader.read_page(bgr, ai_engine, settings)
                    w, lo, who = len(text.split()), text.count("[?]"), "ai:" + ai_engine
                except Exception as e:  # noqa: BLE001 - keep own-model text if the AI call fails
                    ai_problems.append(f"page {page_no}: {e}")
            used.add(who)
            words_total += w
            low_total += lo
            texts.append(text)
            preview = None
            if preview_prefix:
                preview = f"{preview_prefix}_p{page_no}.jpg"
                cv2.imwrite(preview, draw_boxes(own["gray"], own["lines"], own["results"]),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            pages_out.append({"lines": own["page_lines"], "preview": preview, "reader": who})
    reader = "+".join(sorted(used)) if used else "own"
    return {"text": "\n\n".join(texts).strip(), "pages": pages_out, "words": words_total,
            "low_conf": low_total, "reader": reader, "ai_problems": ai_problems}
