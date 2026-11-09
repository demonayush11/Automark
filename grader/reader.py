"""Read a whole answer script (photos / PDF pages) into text. Everything runs on this computer.

Reader modes:
  own    - only our own CNN + LSTM + CTC model
  auto   - own model first; if it is unsure about too many words on a page, TrOCR reads that page
  trocr  - TrOCR reads every page
"""
import re
import cv2
from .htr_model import HandwritingModel
from .segment import load_pages, segment_page, draw_boxes

_model = None
LOW_CONF = 0.5         # words below this are marked [?] for the teacher to check
AUTO_SHARE = 0.25      # auto mode: if more than 25% of a page's words are unsure, TrOCR reads that page

READER_NAMES = {"own": "own model", "trocr": "TrOCR"}


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


def _trocr_read_page(own):
    """TrOCR reads the same lines the own model's cutter found."""
    from . import trocr_reader
    crops = trocr_reader.line_crops(own["gray"], own["lines"])
    results = trocr_reader.read_lines(crops) if crops else []
    lines, words, low = [], 0, 0
    for text, conf in results:
        n = len(text.split())
        words += n
        if conf < trocr_reader.LOW_CONF and text:
            low += n
            text += " [?]"
        lines.append(text)
    return "\n".join(lines), words, low


def _looks_bad(r):
    """Own-model result too poor to trust: nothing found, or many unsure words."""
    if r["words"] == 0:
        return True
    return r["low"] / r["words"] > AUTO_SHARE


def read_script(paths, preview_prefix=None, mode="own"):
    """paths: one student's files in page order.
    Returns dict(text, pages, words, low_conf, reader, problems)."""
    model = get_model()
    pages_out, texts = [], []
    words_total = low_total = 0
    used, problems = set(), []
    page_no = 0
    for path in paths:
        for bgr in load_pages(path):
            page_no += 1
            own = _own_read_page(model, bgr)
            text, w, lo, who = own["text"], own["words"], own["low"], "own"
            if own["lines"] and (mode == "trocr" or (mode == "auto" and _looks_bad(own))):
                try:
                    text, w, lo = _trocr_read_page(own)
                    who = "trocr"
                except Exception as e:  # noqa: BLE001 - keep own-model text if TrOCR is missing/fails
                    if mode == "trocr":
                        raise
                    problems.append(f"page {page_no}: {e}")
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
            "low_conf": low_total, "reader": reader, "problems": problems}
