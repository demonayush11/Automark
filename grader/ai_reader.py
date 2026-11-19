"""
Backup reader: an AI vision model reads the whole page image.
Used when the own model is unsure about many words (messy writing, maths, symbols),
or when the teacher clicks "Read with AI".

Engines: Groq (free, Llama 4 Scout vision model) or Gemini.
"""
import base64
import os

import cv2

from . import marker

GROQ_VISION_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_VISION_MODELS",
    "meta-llama/llama-4-scout-17b-16e-instruct,meta-llama/llama-4-maverick-17b-128e-instruct").split(",") if m.strip()]

PROMPT = """You transcribe a student's handwritten exam answer sheet.
Write out ALL text on the page exactly as the student wrote it.
Rules:
- Keep the original line breaks and order (top to bottom).
- Keep question labels exactly (Q1, Q.1, 1), Ans 2, Sol, etc.).
- Write maths in plain text: 784 x 256 = 200704, 2^8, sqrt(x), a/b.
- Do NOT correct spelling or grammar. Do NOT answer, explain or summarise.
- A word you cannot read: write [?].
- A diagram or figure: write [diagram: a few words saying what it shows].
Output only the transcription."""


def _page_jpeg(bgr, long_side=1600, quality=85):
    h, w = bgr.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def pick_engine(settings):
    """Which AI reads pages: the chosen marking engine if it has a key, else whichever key exists."""
    if settings.get("engine") == "gemini" and settings.get("api_key"):
        return "gemini"
    if settings.get("groq_key"):
        return "groq"
    if settings.get("api_key"):
        return "gemini"
    return None


def read_page(bgr, engine, settings):
    jpg = _page_jpeg(bgr)
    if engine == "groq":
        url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
        content = [{"type": "text", "text": "Transcribe this answer sheet page."},
                   {"type": "image_url", "image_url": {"url": url}}]
        text = marker._call_groq(PROMPT, content, settings["groq_key"], json_mode=False,
                                 models=GROQ_VISION_MODELS)
    elif engine == "gemini":
        from google.genai import types
        text = marker._call_gemini([PROMPT, types.Part.from_bytes(data=jpg, mime_type="image/jpeg")],
                                   settings["api_key"], json_mode=False)
    else:
        raise RuntimeError("Reading with AI needs a Groq or Gemini key (Settings page).")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    return text.strip()
