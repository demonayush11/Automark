# AutoMark — automatic answer-script checker

Teacher uploads a marking scheme, then each student's handwritten answer script (photo or PDF).
The app reads the handwriting, the teacher checks the text, and each answer is marked against the
scheme. Results download as a CSV.

**Everything runs on this computer. No API is called.** Models are downloaded once, then work offline.

## Run it

1. Install **Python 3.10 – 3.12** (python.org). On Windows tick "Add Python to PATH".
2. Double-click **`start_windows.bat`** (Mac/Linux: `./start_mac_linux.sh`).
   First run installs packages (about 1 GB, mostly PyTorch).
3. Open **http://127.0.0.1:5000**.
4. The first time TrOCR or Sentence-BERT is used, its model downloads (TrOCR about 1.3 GB, Sentence-BERT about 90 MB).
   Do this once with internet before the demo.

Try it with the files in `sample/`.

## How to use

1. **New test** → name it → upload the marking scheme (typed PDF / Word / TXT) or paste it.
2. **Check the scheme.** Format:
   ```
   Q1 [3] Define photosynthesis.
   - plants make their own food (1)
   - using sunlight (1)
   - from carbon dioxide and water (1)
   ```
3. **Upload scripts.** "One student" = several pages of one student. "Many students" = one file per student, file name = student name.
4. **Review.** Left: the page with a box on every word (green = sure, orange = unsure). Right: the text.
   Words ending in `[?]` were hard to read — fix them. Buttons **Read with TrOCR** / **Read with own model** re-read the script.
5. **Mark.** Marks per question and per point, with the part of the answer that earned each mark. Change any mark and save.
6. **Download CSV** from the test page.

## Models used (all local)

| Job | Model | Notes |
|---|---|---|
| Read handwriting (main) | **Own CNN + BiLSTM + CTC** (`grader/htr_model.py`) | Weights trained by AutoEx on the IAM dataset. Reads one word at a time. |
| Spelling correction | **Lexicon decoding** (`grader/lexicon.py`) | For each word, scores close English words (58,954-word list + marking-scheme words) with the CTC forward algorithm and word frequency; keeps the most likely. |
| Read handwriting (backup) | **TrOCR** `microsoft/trocr-base-handwritten` (`grader/trocr_reader.py`) | Vision Transformer encoder + text Transformer decoder. Reads one line at a time. Used for pages where the own model is unsure about over 25% of words. |
| Marking (main) | **Sentence-BERT** `all-MiniLM-L6-v2` (`grader/marker.py`) | Turns sentences into 384-number vectors of their meaning. A marking point gets full marks if some part of the answer has cosine similarity ≥ 0.60, half marks at ≥ 0.45. |
| Marking (simple) | Fuzzy keyword matching | No model. Checks key words of each point, allowing small spelling errors. |

Settings → **Reading handwriting**: own model + TrOCR if unsure (default) / own model only / TrOCR only.
Settings → **Marking**: Sentence-BERT (default) / keywords.

## How it works

```
photo / PDF
  │  segment.py   clean page (remove grain/noise, shadows, straighten tilt up to ±15°, erase ruled lines,
  │               ignore binder holes and page-edge marks)
  │               → find text lines (horizontal ink profile)
  │               → find words (gaps between ink blobs, split by 1-D Otsu threshold)
  ▼
word images ──► htr_model.py  OWN MODEL: CNN (5 layers) → BiLSTM (2 layers) → CTC → text + confidence
  │            ──► lexicon.py    dictionary correction ("crwelty" → "cruelty", "seebecause" → "see because")
  │
  │  page with >25% unsure words → line images ──► trocr_reader.py  TrOCR reads each line
  ▼
student text ── teacher corrects ──► marker.py  Sentence-BERT: meaning of answer vs each marking point
                                     → marks per point, per question, total ──► results + CSV
```

| File | What it does |
|---|---|
| `grader/segment.py` | Page cleaning, line and word cutting |
| `grader/htr_model.py` | Own handwriting model, run with NumPy |
| `grader/lexicon.py` | Dictionary correction of each word (CTC forward algorithm + word frequency) |
| `model/english_words.txt` | 58,954 English words with frequencies (from the `wordfreq` project) |
| `evaluate.py` | Accuracy test: 2 samples × 11 kinds of damage, letter and word error rates |
| `grader/trocr_reader.py` | TrOCR backup reader (PyTorch + Hugging Face Transformers) |
| `grader/reader.py` | Runs the readers on all pages of a script |
| `grader/marker.py` | Reads the scheme, marks answers (Sentence-BERT or keywords) |
| `grader/db.py` | SQLite storage and settings in `data/` |
| `app.py` + `templates/` | The teacher website (Flask) |
| `model/htr_weights.npz` | Own model's trained weights (6 MB) |

## Accuracy (own model only, `python evaluate.py`)

Test page: the sample sheet (85 words, known text), as a scan and as a phone-style photo, with 11 kinds of damage.
CER = wrong letters, WER = wrong words (lower is better).

| | Before | After |
|---|---|---|
| Scan, clean | CER 4.8%, WER 19.2% | CER 3.3%, WER 11.5% |
| Photo, clean | CER 9.4%, WER 34.6% | CER 6.6%, WER 23.1% |
| Photo, tilted 4° | CER 63.3%, WER 75.6% | CER 8.6%, WER 28.2% |
| Photo, heavy noise | CER 47.1%, WER 84.6% | CER 6.6%, WER 16.7% |
| Photo, blur + noise | CER 68.6%, WER 96.2% | CER 24.6%, WER 59.0% |
| Average of all 22 | CER 14.2%, WER 33.7% | CER 6.4%, WER 19.1% |

Changes: wider tilt correction (±5° → ±15°, 0.1° steps), denoising of grainy images, dictionary correction.
Correction settings were chosen on 6 separate lines by other writers, not on the test page.
The test page comes from IAM, the dataset the model was trained on, so real students' sheets will score worse.

## Credits

- Own model weights: **AutoEx** — https://github.com/4bdul4ziz/AutoEx (trained on IAM).
- Own model architecture: **SimpleHTR** by Harald Scheidl — https://github.com/githubharald/SimpleHTR
- **TrOCR**: Microsoft — https://huggingface.co/microsoft/trocr-base-handwritten
- Word list: **wordfreq** by Robyn Speer — https://github.com/rspeer/wordfreq
- **Sentence-BERT**: sentence-transformers — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Known limits

- Both reading models learned from IAM (mostly British writers). Indian handwriting, maths and symbols read worse.
  The teacher review step exists for this reason.
- No diagrams. English only.
- TrOCR is slow on a laptop CPU (a page can take tens of seconds).
- Sentence-BERT thresholds (0.60 / 0.45) are starting values; tune them on real marked answers.
- Marking schemes must be typed (or pasted); photos of schemes are not read.
