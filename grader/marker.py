"""
Marking: compare a student's extracted text with the teacher's marking scheme.
Everything runs on this computer; no API is called.

Two marking engines:
  * "similarity" - Sentence-BERT (all-MiniLM-L6-v2) turns each marking point and each part of the
                   student's answer into a vector of 384 numbers that captures its MEANING.
                   If a part of the answer is close in meaning to a point (cosine similarity), the
                   point gets its marks - even when the student used different words.
  * "keywords"   - fuzzy keyword matching against each marking point (no model, very fast).

A marking scheme is stored as JSON:
{
  "questions": [
    {"id": "1", "question": "Define photosynthesis.", "max_marks": 3,
     "points": [{"text": "plants make food using sunlight", "marks": 1, "keywords": ["sunlight"]}, ...]}
  ]
}
"""
import os
import re
from difflib import SequenceMatcher

SBERT_MODEL = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ================================================================ reading the scheme file
def extract_file_text(path):
    """Typed text from a PDF / DOCX / TXT. Returns '' if the file is a scan or photo."""
    low = path.lower()
    if low.endswith(".txt"):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    if low.endswith(".docx"):
        import docx
        d = docx.Document(path)
        rows = [p.text for p in d.paragraphs]
        for t in d.tables:
            for r in t.rows:
                rows.append(" | ".join(c.text for c in r.cells))
        return "\n".join(rows)
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return ""


def parse_scheme_simple(text):
    """Offline parser for a simple typed format, e.g.

        Q1 [3] Define photosynthesis.
        - plants make their own food (1)
        - using sunlight (1)
        - chlorophyll / carbon dioxide and water (1)
    """
    questions, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(?:Q(?:uestion)?\s*)?(\d+[a-z]?)[\).:\s]\s*(?:\[(\d+(?:\.\d+)?)\])?\s*(.*)$", line, re.I)
        if m and not line.startswith(("-", "*", "•")):
            cur = {"id": m.group(1), "question": m.group(3), "max_marks": float(m.group(2) or 0), "points": []}
            questions.append(cur)
            continue
        if cur is not None:
            pm = re.match(r"^[-*•]?\s*(.+?)\s*\((\d+(?:\.\d+)?)\s*(?:marks?|m)?\)\s*$", line, re.I)
            if pm:
                cur["points"].append({"text": pm.group(1), "marks": float(pm.group(2))})
            else:
                cur["points"].append({"text": line.lstrip("-*• "), "marks": 1.0})
    return normalise_scheme({"questions": questions})


def normalise_scheme(data):
    qs = []
    for i, q in enumerate(data.get("questions", []), 1):
        pts = []
        for p in q.get("points", []):
            text = str(p.get("text", "")).strip()
            if not text:
                continue
            kws = p.get("keywords") or _auto_keywords(text)
            pts.append({"text": text, "marks": float(p.get("marks", 1) or 0), "keywords": [str(k) for k in kws]})
        total = sum(p["marks"] for p in pts)
        mx = float(q.get("max_marks") or 0) or total
        qs.append({"id": str(q.get("id") or i), "question": str(q.get("question", "")),
                   "max_marks": mx, "points": pts})
    return {"questions": qs}


STOP = set("a an the of to in on and or is are was were be by for with as at from that this it its into "
           "their they them which who what when where why how can will would should could must has have had "
           "not no do does did so such than then there these those also each any all".split())


def _auto_keywords(text):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower()) if w not in STOP and len(w) > 2]
    words.sort(key=len, reverse=True)
    return words[:4]



# ================================================================ grading
def _clean_results(scheme, results, engine):
    """Make sure every scheme question is present and marks stay within limits."""
    by_id = {str(r.get("id")): r for r in results}
    out = []
    for q in scheme["questions"]:
        r = by_id.get(q["id"], {})
        pts = r.get("points") or [{"text": p["text"], "awarded": 0, "max": p["marks"], "evidence": ""} for p in q["points"]]
        for p in pts:
            p["max"] = float(p.get("max", 0) or 0)
            p["awarded"] = max(0.0, min(float(p.get("awarded", 0) or 0), p["max"] or 1e9))
        awarded = float(r.get("awarded", sum(p["awarded"] for p in pts)) or 0)
        awarded = round(max(0.0, min(awarded, q["max_marks"])) * 2) / 2
        out.append({"id": q["id"], "question": q["question"], "awarded": awarded, "max": q["max_marks"],
                    "points": pts, "feedback": r.get("feedback", "")})
    total = sum(r["awarded"] for r in out)
    maximum = sum(r["max"] for r in out)
    return {"engine": engine, "total": total, "max": maximum, "questions": out}


def _norm_words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("[?]", ""))


def _fuzzy_in(keyword, words, cutoff=0.75):
    """Is keyword (possibly misread) present among the student's words?"""
    k = keyword.lower()
    best, hit = 0.0, ""
    for w in words:
        r = SequenceMatcher(None, k, w).ratio()
        if r > best:
            best, hit = r, w
    return best >= cutoff, hit


def split_answers(student_text, ids):
    """Cut student text into answers using labels like Q1 / 1) / Ans 2. Falls back to whole text."""
    # "Q1", "Q 2a", "Ans 3", "Answer 4:" (label word present) or bare "1)", "2a." (needs punctuation)
    pattern = re.compile(r"(?im)^\s*(?:(?:q(?:uestion)?|ans(?:wer)?)\s*[.\-]?\s*(\d+\s?[a-z]?)\b"
                         r"|(\d+\s?[a-z]?)\s*[\).:\-])")
    marks = [(m.start(), re.sub(r"\s", "", m.group(1) or m.group(2)).lower()) for m in pattern.finditer(student_text)]
    marks = [(pos, i) for pos, i in marks if i in ids]
    if not marks:
        return {}
    out = {}
    for n, (pos, qid) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(student_text)
        out[qid] = out.get(qid, "") + "\n" + student_text[pos:end]
    return out


def grade_keywords(scheme, student_text):
    ids = {q["id"].lower() for q in scheme["questions"]}
    pieces = split_answers(student_text, ids)
    results = []
    for q in scheme["questions"]:
        text = pieces.get(q["id"].lower(), student_text if not pieces else "")
        words = _norm_words(text)
        pts = []
        for p in q["points"]:
            kws = p.get("keywords") or _auto_keywords(p["text"])
            found = [(k, _fuzzy_in(k, words)) for k in kws]
            hits = [(k, w) for k, (ok, w) in found if ok]
            share = len(hits) / len(kws) if kws else 0
            awarded = p["marks"] if share >= 0.5 else (p["marks"] / 2 if share > 0 else 0)
            pts.append({"text": p["text"], "awarded": round(awarded * 2) / 2, "max": p["marks"],
                        "evidence": ", ".join(w for _, w in hits)})
        results.append({"id": q["id"], "points": pts,
                        "awarded": sum(x["awarded"] for x in pts),
                        "feedback": "Keyword match."})
    return _clean_results(scheme, results, engine="keywords")



# ================================================================ meaning-based marking (Sentence-BERT)
FULL_SIM = 0.60     # similarity at or above this: full marks for the point
HALF_SIM = 0.45     # at or above this: half marks

_sbert = None


def _sbert_model():
    global _sbert
    if _sbert is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("Similarity marking needs sentence-transformers. "
                               "Run: pip install -r requirements.txt") from e
        try:
            _sbert = SentenceTransformer(SBERT_MODEL)
        except OSError as e:
            raise RuntimeError(f"Could not load Sentence-BERT model '{SBERT_MODEL}'. The first use needs "
                               f"internet to download it (about 90 MB). Details: {e}") from e
    return _sbert


def answer_chunks(text, window=12, step=6):
    """Pieces of the answer to compare with each marking point:
    every sentence/line, plus overlapping windows of 12 words (a point is often half a sentence)."""
    clean = text.replace("[?]", " ")
    chunks = [c.strip() for c in re.split(r"[\n.;!?]+", clean) if len(c.split()) >= 2]
    words = clean.split()
    for i in range(0, max(1, len(words) - window + 1), step):
        piece = " ".join(words[i:i + window])
        if len(piece.split()) >= 2:
            chunks.append(piece)
    seen, out = set(), []
    for c in chunks:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def grade_similarity(scheme, student_text):
    model = _sbert_model()
    ids = {q["id"].lower() for q in scheme["questions"]}
    pieces = split_answers(student_text, ids)
    results = []
    for q in scheme["questions"]:
        text = pieces.get(q["id"].lower(), student_text if not pieces else "")
        chunks = answer_chunks(text)
        pts = []
        if chunks and q["points"]:
            point_vecs = model.encode([p["text"] for p in q["points"]], normalize_embeddings=True)
            chunk_vecs = model.encode(chunks, normalize_embeddings=True)
            sims = point_vecs @ chunk_vecs.T          # cosine similarity (vectors have length 1)
        for i, p in enumerate(q["points"]):
            if not chunks:
                pts.append({"text": p["text"], "awarded": 0.0, "max": p["marks"], "evidence": ""})
                continue
            j = int(sims[i].argmax())
            best = float(sims[i][j])
            awarded = p["marks"] if best >= FULL_SIM else (p["marks"] / 2 if best >= HALF_SIM else 0)
            pts.append({"text": p["text"], "awarded": round(awarded * 2) / 2, "max": p["marks"],
                        "evidence": f"{chunks[j]}  (similarity {best:.2f})" if best >= HALF_SIM else "",
                        "similarity": round(best, 3)})
        results.append({"id": q["id"], "points": pts, "awarded": sum(x["awarded"] for x in pts),
                        "feedback": f"Meaning match with Sentence-BERT (full marks at similarity "
                                    f"{FULL_SIM:.2f}+, half at {HALF_SIM:.2f}+)."})
    return _clean_results(scheme, results, engine="similarity")


def scheme_to_text(scheme):
    """Turn scheme JSON into the simple editable format used on the 'Edit scheme' page."""
    lines = []
    for q in scheme.get("questions", []):
        mx = q["max_marks"]
        lines.append(f"Q{q['id']} [{mx:g}] {q['question']}".rstrip())
        for p in q["points"]:
            lines.append(f"- {p['text']} ({p['marks']:g})")
        lines.append("")
    return "\n".join(lines).strip()
