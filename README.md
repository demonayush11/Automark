# AutoMark — automatic answer-script checker

Teacher uploads a marking scheme, then each student's handwritten answer script (photo or PDF).
The app reads the handwriting with its **own trained model**, the teacher checks the text,
and the answers are marked against the scheme. Results download as a CSV.

## Run it

1. Install **Python 3.10 – 3.13** (python.org). On Windows tick "Add Python to PATH".
2. Unzip this folder.
3. Double-click **`start_windows.bat`** (Mac/Linux: run `./start_mac_linux.sh`).
   First run installs everything (a few minutes).
4. Open **http://127.0.0.1:5000** in your browser.
5. **Settings → paste a free Groq API key** (from https://console.groq.com/keys, no credit card) and select **Groq**.
   Gemini (https://aistudio.google.com/apikey) also works. Without any key the app uses offline keyword marking.

Try it with the files in `sample/`.

## How to use

1. **New test** → name it → upload the marking scheme (PDF / Word / TXT, or a photo if Gemini is on) or paste it.
2. **Check the scheme.** Format:
   ```
   Q1 [3] Define photosynthesis.
   - plants make their own food (1)
   - using sunlight (1)
   - from carbon dioxide and water (1)
   ```
3. **Upload scripts.** "One student" = several pages of one student. "Many students" = one file per student, file name = student name.
4. **Review.** Left: the page with a box on every word (green = sure, orange = unsure). Right: the text. Words ending in `[?]` were hard to read — fix them.
   Buttons **Read with AI** / **Read with own model** re-read the script either way.
5. **Mark.** See marks per question and per point, with the student words that earned each mark. Change any mark and save.
6. **Download CSV** from the test page.

## Who reads the handwriting (Settings → Reading handwriting)

| Setting | What happens |
|---|---|
| **Own model, AI if unsure** (default) | Own model reads each page. If more than 25% of a page's words are unsure, that page is read again by AI. |
| Own model only | Only the project's model. Best for showing the model in a viva. |
| AI only | AI vision reads every page: Groq `llama-4-scout` (free) or Gemini. |

The own model is good at clear English handwriting. AI is needed for messy writing, maths and symbols.

## How it works (for your report / viva)

```
photo / PDF
  │  segment.py   clean page (remove shadows, straighten tilt, erase ruled lines,
  │               ignore binder holes and page-edge marks)
  │               → find text lines (horizontal ink profile)
  │               → find words (gaps between ink blobs, split by 1-D Otsu threshold)
  ▼
word images
  │  htr_model.py  OWN MODEL: CNN (5 layers) → bidirectional LSTM (2 layers) → CTC decoding
  │                input 128×32 grayscale word → text + confidence
  ▼
  │  (auto mode) page with >25% unsure words → ai_reader.py reads the page image instead
  ▼
student text  ──  teacher corrects  ──►  marker.py
                                           • Groq / Gemini: AI compares meaning with each marking point
                                           • Offline: fuzzy keyword match per marking point
                                         → marks per point, per question, total
```

| File | What it does |
|---|---|
| `grader/segment.py` | Page cleaning, line and word cutting |
| `grader/htr_model.py` | The handwriting model, run with NumPy (no TensorFlow needed) |
| `grader/reader.py` | Runs segmentation + model on all pages; sends unsure pages to AI |
| `grader/ai_reader.py` | AI vision reading of a page (Groq Llama 4 Scout or Gemini) |
| `grader/marker.py` | Reads the scheme, marks answers (Groq, Gemini or offline) |
| `grader/db.py` | SQLite storage in `data/` |
| `app.py` + `templates/` | The teacher website (Flask) |
| `model/htr_weights.npz` | Trained model weights (6 MB) |

**Confidence:** for each word, the lowest probability among its letters. Below 0.5 → marked `[?]`.

**AI models:** Groq uses `llama-3.3-70b-versatile` (falls back to `openai/gpt-oss-120b`, then `llama-3.1-8b-instant`
when busy). Gemini uses `gemini-3.8-flash`. Change with environment variables `GROQ_MODEL` / `GEMINI_MODEL`.
Groq reads only typed marking schemes; use Gemini for a photo of the scheme.

## Credits

- Handwriting model weights: **AutoEx** by Abdul A. and Samik S. — https://github.com/4bdul4ziz/AutoEx
  (trained on the IAM handwriting dataset).
- Model architecture: **SimpleHTR** by Harald Scheidl — https://github.com/githubharald/SimpleHTR
- Changes in this project: model rewritten to run in NumPy (original needed old TensorFlow on a Mac),
  new page/line/word segmentation, ruled-line removal, confidence flags, marking-scheme parsing,
  Groq / Gemini / offline marking, and the whole web app. AutoEx's Apple-Vision part and R interface were removed.

## Known limits

- The model learned from IAM — English handwriting from mostly British writers. Expect more reading errors
  on other handwriting styles. The teacher review step exists for this reason.
- It reads English letters, digits and basic punctuation only. No diagrams, maths symbols or other languages.
- Words written very close together may be read as one word.
- Text written straight over ruled lines reads worse than text between lines.
- Offline marking only checks key words. It cannot judge meaning. Use Groq or Gemini for real marking.
- Groq free tier: about 1,000 marking requests per day for the main model.
