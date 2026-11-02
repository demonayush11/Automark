"""
Marking: compare a student's extracted text with the teacher's marking scheme.

Three marking engines:
  * "groq"    - free Groq API (open models like Llama 3.3 70B) awards marks per point.
  * "gemini"  - Google's Gemini model reads the scheme + student text and awards marks per point.
  * "offline" - no internet needed: fuzzy keyword matching against each marking point.
                Less smart, but works for demos and when the API is down.

A marking scheme is stored as JSON:
{
  "questions": [
    {"id": "1", "question": "Define photosynthesis.", "max_marks": 3,
     "points": [{"text": "plants make food using sunlight", "marks": 1, "keywords": ["sunlight"]}, ...]}
  ]
}
"""
import json
import os
import re
from difflib import SequenceMatcher

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,llama-3.1-8b-instant").split(",") if m.strip()]


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


SCHEME_PROMPT = """You convert a teacher's marking scheme into JSON.
Return ONLY JSON in this exact shape:
{"questions":[{"id":"1","question":"<question text or short title>","max_marks":<number>,
  "points":[{"text":"<one thing the answer must contain>","marks":<number>,"keywords":["<1-4 key words>"]}]}]}
Rules:
- One entry per question / sub-question (use ids like "1", "2a", "2b").
- Split each model answer into separate marking points. Point marks must add up to max_marks.
- If the scheme gives only a model answer and total marks, split it into sensible points yourself.
- Keep wording close to the scheme. Do not invent questions."""


def parse_scheme_with_gemini(path, api_key):
    text = extract_file_text(path)
    parts = [SCHEME_PROMPT]
    if text.strip():
        parts.append("MARKING SCHEME:\n" + text)
    else:   # scanned PDF or photo: let Gemini look at the file itself
        parts.append("The marking scheme is in the attached file.")
        parts.append(_file_part(path))
    data = _call_gemini(parts, api_key)
    return normalise_scheme(data)


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
GRADE_PROMPT = """You are an exam examiner. Mark one student's answer script using the marking scheme.

IMPORTANT about the student text:
- It was produced by a handwriting-recognition program, NOT typed by the student.
- It contains reading mistakes: wrong letters, merged words, missing punctuation.
- Words ending in [?] were hard to read.
- Judge what the student MEANT. Do not cut marks for spelling or recognition errors.
- Answers may be in any order. Students may label them "Q1", "1)", "Ans 1", etc., or not at all.

Marking rules:
- Award marks only for points in the scheme. Never exceed a question's max_marks.
- Partial marks are allowed only where a point is partly covered (use halves).
- If an answer is missing, award 0.

Return ONLY JSON:
{"results":[{"id":"<question id>","awarded":<number>,"max":<number>,
  "points":[{"text":"<scheme point>","awarded":<number>,"max":<number>,"evidence":"<student words that earned it, or empty>"}],
  "feedback":"<one short sentence for the teacher>"}]}
"""


def grade_with_gemini(scheme, student_text, api_key):
    parts = [GRADE_PROMPT,
             "MARKING SCHEME (JSON):\n" + json.dumps(scheme, ensure_ascii=False),
             "STUDENT TEXT:\n" + student_text]
    data = _call_gemini(parts, api_key)
    return _clean_results(scheme, data.get("results", []), engine="gemini")


def parse_scheme_with_groq(path, api_key):
    text = extract_file_text(path)
    if not text.strip():
        raise ValueError("Groq can only read typed PDF / Word / TXT schemes. "
                         "Paste the scheme as text, or switch to Gemini in Settings for photos/scans.")
    return normalise_scheme(_call_groq(SCHEME_PROMPT, "MARKING SCHEME:\n" + text, api_key))


def grade_with_groq(scheme, student_text, api_key):
    user = ("MARKING SCHEME (JSON):\n" + json.dumps(scheme, ensure_ascii=False) +
            "\n\nSTUDENT TEXT:\n" + student_text)
    data = _call_groq(GRADE_PROMPT, user, api_key)
    return _clean_results(scheme, data.get("results", []), engine="groq")


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


def grade_offline(scheme, student_text):
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
                        "feedback": "Keyword match (offline mode)."})
    return _clean_results(scheme, results, engine="offline")


# ================================================================ Gemini plumbing
def _file_part(path):
    from google.genai import types
    ext = path.lower().rsplit(".", 1)[-1]
    mime = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp"}.get(ext, "application/octet-stream")
    with open(path, "rb") as fh:
        return types.Part.from_bytes(data=fh.read(), mime_type=mime)


# If the main model is busy, these are tried next (lighter models are usually less busy).
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",") if m.strip()]
BUSY_CODES = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand")


def _call_gemini(parts, api_key, json_mode=True):
    """Call Gemini. On 'busy' errors: wait and retry, then try the fallback models."""
    if not api_key:
        raise RuntimeError("No Gemini API key. Add it on the Settings page, or use offline marking.")
    import time
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0,
                                         response_mime_type="application/json" if json_mode else "text/plain")
    models = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None
    for model in models:
        for attempt in range(3):                       # 3 tries per model: wait 2s, 5s between
            try:
                resp = client.models.generate_content(model=model, contents=parts, config=config)
                return _parse_json(resp.text or "") if json_mode else (resp.text or "")
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                last_error = e
                if "404" in msg or "NOT_FOUND" in msg:  # model name not available: try next model
                    break
                if not any(code in msg for code in BUSY_CODES):
                    raise                               # real error (bad key, bad request): stop
                if attempt < 2:
                    time.sleep((2, 5)[attempt])
    raise RuntimeError("Gemini is busy right now (tried " + ", ".join(models) +
                       "). Wait a minute and click Mark again, or choose offline marking. "
                       f"Last error: {last_error}")


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("The AI did not return valid JSON:\n" + text[:500])


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


# ================================================================ Groq plumbing (no extra library needed)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq(system_prompt, user_text, api_key, json_mode=True, models=None):
    """Call Groq's chat API. user_text may be a string or a list of content parts (text + images).
    On busy / rate-limit errors: wait, retry, then try fallback models."""
    if not api_key:
        raise RuntimeError("No Groq API key. Add it on the Settings page, or use offline marking.")
    import time
    import urllib.error
    import urllib.request
    models = models or [GROQ_MODEL] + [m for m in GROQ_FALLBACK_MODELS if m != GROQ_MODEL]
    last_error = ""
    for model in models:
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_text}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(3):
            req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
                "User-Agent": "AutoMark/1.0",
            })
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"] or ""
                return _parse_json(content) if json_mode else content
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:300]
                last_error = f"{e.code} {detail}"
                if e.code in (401, 403):
                    raise RuntimeError("Groq rejected the API key. Check it on the Settings page. " + detail)
                if e.code in (400, 404) and ("model" in detail.lower() or e.code == 404):
                    break                                   # model not available: try the next one
                if e.code in (429, 500, 502, 503):
                    wait = e.headers.get("retry-after")
                    time.sleep(min(float(wait), 20) if wait and wait.replace(".", "").isdigit() else (3, 8)[min(attempt, 1)])
                    continue
                raise RuntimeError(f"Groq error {last_error}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"Cannot reach Groq (check internet): {e.reason}")
    raise RuntimeError("Groq is busy or the daily free limit is used up (tried " + ", ".join(models) +
                       "). Wait a minute and try again, or use another engine. Last error: " + last_error)
