"""
Collect training data for fine-tuning (see finetune.py).

After the teacher has corrected the text of a script, "Save as training data" stores every word
image the model saw, labelled with the teacher's corrected spelling:

    data/training/s12_0007.png            one word image (exactly what the model reads)
    data/training/labels.tsv              file <TAB> correct text <TAB> student

How a word image is matched to its corrected text:
  1. The page is cut into lines and words again and read again (same as when it was uploaded),
     which gives the original reading of every word box.
  2. Lines: the original lines are paired with the corrected lines in order. If that fails for
     most lines (e.g. the text was typed as one paragraph), the whole page is aligned at once.
  3. Words: original and corrected text are aligned letter by letter; each word box gets the
     part of the corrected text that lines up with its original reading.

Also used by make_dataset.py (photos + typed text files -> training data, e.g. on Colab).
  Words still marked [?] (not checked by the teacher) and empty labels are skipped.
"""
import csv
import difflib
import io
import os
import zipfile

import cv2

from . import db
from .reader import LOW_CONF, get_model
from .segment import load_pages, segment_page

TRAIN_DIR = os.path.join(db.DATA_DIR, "training")
LABELS = os.path.join(TRAIN_DIR, "labels.tsv")
MAX_LEN = 32                                    # the model has 32 time steps


def _read_labels():
    if not os.path.exists(LABELS):
        return []
    with open(LABELS, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_labels(rows):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(LABELS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "text", "student"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def stats():
    rows = _read_labels()
    return {"words": len(rows), "students": len({r["student"] for r in rows})}


def _char_map(orig, corrected):
    """For each character position in `orig`, the matching position in `corrected`.
    m.inserted[j] is True for letters of `corrected` that have no counterpart in `orig`
    (e.g. a first letter the model missed)."""
    m = _Map([0] * (len(orig) + 1))
    m.inserted = [False] * len(corrected)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, orig, corrected, autojunk=False).get_opcodes():
        if op == "insert":
            for j in range(j1, j2):
                m.inserted[j] = True
        for k in range(i1, i2 + 1):
            if op == "equal":
                m[k] = j1 + (k - i1)
            elif op == "replace":
                m[k] = j1 + round((k - i1) * (j2 - j1) / max(1, i2 - i1))
            else:                     # delete: collapses onto j1 ("insert" has no orig chars)
                m[k] = j1
    return m


class _Map(list):
    inserted = None


def _line_pairs(orig_lines, new_lines):
    """Pair original lines with corrected lines (same order; unequal stretches are skipped)."""
    pairs = []
    sm = difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            pairs += list(zip(range(i1, i2), range(j1, j2)))
    return pairs


def read_boxes(pages_bgr, model):
    """Cut pages into lines/words and read them. Returns (boxes_by_line, orig_lines) where
    boxes_by_line[i] = [(start, end, word_image), ...] = position of each word in orig_lines[i]."""
    boxes_by_line, orig_lines = [], []
    for bgr in pages_bgr:
        _, lines, _ = segment_page(bgr)
        for line in lines:
            heights = sorted(w["img"].shape[0] for w in line)
            tall = 2.0 * heights[len(heights) // 2]
            reads = model.read_words([w["img"] for w in line])
            parts, spans, pos = [], [], 0
            for (word, conf), w in zip(reads, line):
                token = "[?]" if not word else (word + "[?]" if conf < LOW_CONF else word)
                if len(line) > 2 and w["img"].shape[0] > tall:
                    token = "[?]"        # crop covers two lines: never used for training
                elif (w["img"] < 128).mean() < 0.02:
                    token = "[?]"        # (almost) no dark ink: a blank or faint crop
                spans.append((pos, pos + len(token), w["img"]))
                parts.append(token)
                pos += len(token) + 1
            boxes_by_line.append(spans)
            orig_lines.append(" ".join(parts))
    return boxes_by_line, orig_lines


def _similar(a, b):
    a, b = a.lower().replace("[?]", ""), b.lower()
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def _label(new, m, a, b, chars, reading=None, min_similarity=0.0):
    start, end = m[a], m[b]
    # the model often misses a word's first/last letters; the corrected text has them as
    # "inserted" letters right next to the word: include them (stop at spaces)
    while start > 0 and m.inserted[start - 1] and not new[start - 1].isspace():
        start -= 1
    while end < len(new) and m.inserted[end] and not new[end].isspace():
        end += 1
    label = new[start:end].strip()
    if len(label) == 1 and label.isalpha() and label not in "aAI":
        return None              # a lone letter is usually a mark or a piece of a word
    if not label or "[?]" in label or len(label) > MAX_LEN or any(c not in chars for c in label):
        return None
    if min_similarity and reading is not None and _similar(reading, label) < min_similarity:
        return None              # the text here probably does not match this word image
    return label


def label_boxes(boxes_by_line, orig_lines, correct_text, chars, min_similarity=0.0):
    """Match every word image to its correct text. Returns (list of (image, label), skipped).
    1st try: pair lines in order (when the correct text keeps the page's line breaks).
    Fallback: if most lines cannot be paired (e.g. text typed as one paragraph), align the whole
    page as one long string instead.
    min_similarity > 0 skips a word when the model's reading and the label share too few letters
    (protects against text files that are not an exact copy of the page)."""
    new_lines = [ln.strip() for ln in correct_text.splitlines() if ln.strip()]
    pairs = _line_pairs(orig_lines, new_lines)
    n_boxes = sum(len(b) for b in boxes_by_line)
    paired_boxes = sum(len(boxes_by_line[i]) for i, _ in pairs)
    out, skipped = [], 0
    if n_boxes and paired_boxes >= 0.8 * n_boxes:
        paired = set()
        for i, j in pairs:
            paired.add(i)
            new = new_lines[j]
            m = _char_map(orig_lines[i], new)
            for a, b, img in boxes_by_line[i]:
                lab = _label(new, m, a, b, chars, orig_lines[i][a:b], min_similarity)
                if lab is None:
                    skipped += 1
                else:
                    out.append((img, lab))
        skipped += sum(len(boxes_by_line[i]) for i in range(len(boxes_by_line)) if i not in paired)
        return out, skipped
    # whole-page alignment
    orig, spans, offset = [], [], 0
    for line, boxes in zip(orig_lines, boxes_by_line):
        spans += [(offset + a, offset + b, img, line[a:b]) for a, b, img in boxes]
        orig.append(line)
        offset += len(line) + 1
    orig, new = " ".join(orig), " ".join(new_lines)
    m = _char_map(orig, new)
    for a, b, img, reading in spans:
        lab = _label(new, m, a, b, chars, reading, min_similarity)
        if lab is None:
            skipped += 1
        else:
            out.append((img, lab))
    return out, skipped


def export_script(script, chars):
    """Save the word images of one corrected script (app button). Returns (saved, skipped)."""
    model = get_model()
    pages = [bgr for name in script["files"] for bgr in load_pages(os.path.join(db.UPLOAD_DIR, name))]
    boxes_by_line, orig_lines = read_boxes(pages, model)
    pairs, skipped = label_boxes(boxes_by_line, orig_lines, script["text"], chars)

    rows = [r for r in _read_labels() if not r["file"].startswith(f"s{script['id']}_")]
    for f in os.listdir(TRAIN_DIR) if os.path.isdir(TRAIN_DIR) else []:
        if f.startswith(f"s{script['id']}_"):
            os.remove(os.path.join(TRAIN_DIR, f))
    os.makedirs(TRAIN_DIR, exist_ok=True)
    for n, (img, label) in enumerate(pairs):
        fname = f"s{script['id']}_{n:04d}.png"
        cv2.imwrite(os.path.join(TRAIN_DIR, fname), img)
        rows.append({"file": fname, "text": label, "student": script["student"]})
    _write_labels(rows)
    return len(pairs), skipped


def zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.isdir(TRAIN_DIR):
            for f in sorted(os.listdir(TRAIN_DIR)):
                z.write(os.path.join(TRAIN_DIR, f), f)
    return buf.getvalue()
