# AutoMark — automatic answer-script checker

Teacher uploads a marking scheme, then each student's handwritten answer script (photo or PDF).
The app reads the handwriting, the teacher checks the text, and each answer is marked against the
scheme. Results download as a CSV.

**Everything runs on this computer. No API is called.** Models are downloaded once, then work offline.

## Run it

1. Install **Python 3.10 – 3.12** (python.org). On Windows tick "Add Python to PATH".
2. Double-click **`start_windows.bat`** (Mac/Linux: `./start_mac_linux.sh`).
   First run installs packages (about 1 GB; Sentence-BERT needs PyTorch).
3. Open **http://127.0.0.1:5000**.
4. The first time Sentence-BERT marking is used, its model downloads (about 90 MB).
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
   Words ending in `[?]` were hard to read — fix them. A `[?]` on its own means the model saw ink but could not read it.
   If the photo is hard to read, a yellow or red banner explains why (see **Photo quality check** below).
5. **Mark.** Marks per question and per point, with the part of the answer that earned each mark. Change any mark and save.
6. **Download CSV** from the test page.

## Models used (all local)

| Job | Model | Notes |
|---|---|---|
| Read handwriting (main) | **Own CNN + BiLSTM + CTC** (`grader/htr_model.py`) | Weights trained by AutoEx on the IAM dataset. Reads one word at a time. |
| Spelling correction | **Lexicon decoding** (`grader/lexicon.py`) | For each word, scores close English words (58,954-word list + marking-scheme words) with the CTC forward algorithm and word frequency; keeps the most likely. |
| Marking (main) | **Sentence-BERT** `all-MiniLM-L6-v2` (`grader/marker.py`) | Turns sentences into 384-number vectors of their meaning. A marking point gets full marks if some part of the answer has cosine similarity ≥ 0.60, half marks at ≥ 0.45. |
| Marking (simple) | Fuzzy keyword matching | No model. Checks key words of each point, allowing small spelling errors. |

There is one reader: our own model. There is no second reader and no API: if a page is too hard to
read reliably, the teacher is asked for a clearer photo.

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
  │
  │            ──► reader.py     photo quality check → OK / Check / Re-upload
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
| `grader/training_data.py` | "Save as training data": word images + corrected text for fine-tuning |
| `make_dataset.py` | Photos + typed text files → training data (no app needed) |
| `finetune.py` | Fine-tuning script (PyTorch; run on Google Colab) |
| `AutoMark_finetune.ipynb` | Colab notebook that runs `finetune.py` step by step |
| `grader/reader.py` | Reads all pages of a script and runs the photo quality check (`page_quality`) |
| `grader/marker.py` | Reads the scheme, marks answers (Sentence-BERT or keywords) |
| `grader/db.py` | SQLite storage and settings in `data/` |
| `app.py` + `templates/` | The teacher website (Flask) |
| `model/htr_weights.npz` | Own model's trained weights (6 MB) |

## Photo quality check

After reading, every page gets a level (code: `page_quality` in `grader/reader.py`; all thresholds are
named constants at the top of that file):

| Check | Warn (yellow) | Bad (red) |
|---|---|---|
| Unsure words (below confidence 0.5, or ink the model could not read at all) | 15% or more | 25% or more |
| No text lines or no words found | | always |
| Tilt: correction searches up to ±15°; a page needing 14° or more may be tilted further | 14° or more | |
| Blur: variance of the Laplacian of the cleaned page | below 18 | |
| Words per line: median below 3 on a page with 5+ lines (words were probably missed) | yes | |

A script's level is the worst of its pages. It is saved in the database, so it survives a reload.

What the teacher sees:
- **Students list:** a Photo column with **OK / Check / Re-upload** for every student, so after a
  "Many students" upload it is clear at once whose photo needs replacing. "Mark all" skips Re-upload scripts.
- **Check (warn):** yellow banner with the reasons and retake tips. The teacher can still mark.
- **Re-upload (bad):** red banner; **Mark** is disabled. Two options: **Re-upload this script**
  (replaces this student's pages and reads them again) or **Mark anyway** (asks for confirmation).
- Retake tips: lay the page flat, even light with no shadow, camera straight above the page,
  fill the frame, hold steady, or use a scan.

The blur threshold was calibrated with `evaluate.py`: the badly read photos (blur 3, blur + noise)
score 5–12; every well-read case scores 25 or more. `evaluate.py` also checks that a photo tilted 20°
and a blank page are flagged.

## Fine-tuning the model on your own students' handwriting

The model that is fine-tuned is **our own reader (CNN + BiLSTM + CTC)**, starting from AutoEx's IAM weights.
It learned from British writers; fine-tuning on your class's handwriting makes it read that better.

**Training data — either:**
- **Directly from photos (no app needed):** make `pages.zip` with each answer-sheet photo/PDF and a `.txt`
  of the same name holding the exact text on it (`aditya_p1.jpg` + `aditya_p1.txt`; the name before `_`
  is the student). `make_dataset.py` cuts the pages into words and labels them. Aim for 1,000+ words
  from 10+ students.
- **From the app:** correct a student's text, click **Save as training data**, then
  Settings → **Download training_data.zip**.

**Train on Google Colab** (free GPU): push this folder to GitHub, open https://colab.research.google.com →
File → Upload notebook → `AutoMark_finetune.ipynb` → Runtime → Change runtime type → **T4 GPU** → run the cells.
The notebook downloads the code from GitHub; you upload only `pages.zip` or `training_data.zip`.

**Install the new weights:** back up `model/htr_weights.npz`, replace it with the downloaded
`htr_weights_finetuned.npz` (renamed to `htr_weights.npz`), restart the app.

`finetune.py` rebuilds the same network in PyTorch (checked: same output as the app's NumPy model, max
difference 0.00002), starts from the current weights, keeps 10% of the students aside as a test set,
trains with CTC loss, a small learning rate (0.0001) and random image changes (stretch, slant,
stroke thickness), keeps the version with the lowest test error, and saves it in the format the app reads.

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

Quality check on the same 22 cases: the 2 badly read cases (letter errors above 12%) are flagged,
the other 20 are OK, with no false alarms.

Changes: wider tilt correction (±5° → ±15°, 0.1° steps), denoising of grainy images, dictionary correction.
Correction settings were chosen on 6 separate lines by other writers, not on the test page.
The test page comes from IAM, the dataset the model was trained on, so real students' sheets will score worse.

## Credits

- Own model weights: **AutoEx** — https://github.com/4bdul4ziz/AutoEx (trained on IAM).
- Own model architecture: **SimpleHTR** by Harald Scheidl — https://github.com/githubharald/SimpleHTR
- Word list: **wordfreq** by Robyn Speer — https://github.com/rspeer/wordfreq
- **Sentence-BERT**: sentence-transformers — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Known limits

- Both reading models learned from IAM (mostly British writers). Indian handwriting, maths and symbols read worse.
  The teacher review step exists for this reason.
- No diagrams. English only.
- Sentence-BERT thresholds (0.60 / 0.45) are starting values; tune them on real marked answers.
- Marking schemes must be typed (or pasted); photos of schemes are not read.
