"""
Accuracy test for the reading pipeline (our own model) and for the photo quality check.

Takes the two sample sheets (photo + scan), makes damaged copies (blur, noise, low resolution,
shadow, tilt, JPEG compression), reads each one and compares with the correct text.

  CER = character error rate = wrong/missing/extra letters  / total letters
  WER = word error rate      = wrong/missing/extra words    / total words
  (lower is better; capitals and punctuation are ignored)

It also runs the quality check (reader.page_quality) on every case and reports whether it flags
the cases that were read badly (CER > 12%), plus two pages that must always be flagged:
a photo tilted 20 degrees and a blank white page.

Run:  python evaluate.py              -> prints a table and writes evaluation_results.csv
      python evaluate.py --no-lexicon -> same, with dictionary correction switched off
"""
import csv
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grader import reader                      # noqa: E402
from grader.segment import load_pages          # noqa: E402

# correct text of both sample sheets (IAM page r06-137)
TRUTH = """The doorman turned his attention
to the next red-eyed emerger from the
dark; and we went on together to
the station, the children silent because
of the cruelty of the world. Finally Catherine
said, her eyes wet again: 'I think its all
absolutely beastly, and I can't bear to think
about it.' And Philip said: 'But we've got
to think about it, don't you see, because
if we don't it'll just go on and on, don't
you see?'"""

SAMPLES = {"photo": os.path.join(HERE, "sample", "sample_answer_photo.jpg"),
           "scan": os.path.join(HERE, "sample", "sample_answer_scan.pdf")}


# ---------------------------------------------------------------- damage functions
def blur(img, s):
    return cv2.GaussianBlur(img, (0, 0), s)


def noise(img, s):
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def lowres(img, f):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def shadow(img, strength):
    h, w = img.shape[:2]
    ramp = np.linspace(1 - strength, 1, w)[None, :, None]
    return np.clip(img * ramp, 0, 255).astype(np.uint8)


def tilt(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


DAMAGE = [
    ("clean", lambda i: i),
    ("blur_1.5", lambda i: blur(i, 1.5)),
    ("blur_3", lambda i: blur(i, 3)),
    ("noise_20", lambda i: noise(i, 20)),
    ("noise_45", lambda i: noise(i, 45)),
    ("lowres_0.5", lambda i: lowres(i, 0.5)),
    ("lowres_0.3", lambda i: lowres(i, 0.3)),
    ("shadow_0.5", lambda i: shadow(i, 0.5)),
    ("tilt_4deg", lambda i: tilt(i, 4)),
    ("jpeg_q10", lambda i: jpeg(i, 10)),
    ("blur3+noise20", lambda i: noise(blur(i, 3), 20)),
]


# ---------------------------------------------------------------- scoring
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def normalise(text):
    text = text.replace("[?]", " ").lower().replace("\n", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]", "", text).split())


def cer_wer(truth, out):
    t, o = normalise(truth), normalise(out)
    cer = edit_distance(t, o) / max(1, len(t))
    wer = edit_distance(t.split(), o.split()) / max(1, len(t.split()))
    return cer, wer


SHOULD_FLAG_CER = 12.0   # a case "should be flagged" if more than 12% of letters are wrong


def run(lexicon=True, out_csv=os.path.join(HERE, "evaluation_results.csv")):
    from grader import htr_model
    htr_model.USE_LEXICON = lexicon
    model = reader.get_model()
    rows = []
    print(f"{'sample':6} {'damage':14} {'CER':>6} {'WER':>6} {'unsure':>7}  quality  reasons")
    for sample, path in SAMPLES.items():
        page = load_pages(path)[0]
        for name, fn in DAMAGE:
            r = reader._own_read_page(model, fn(page))
            q = reader.page_quality(r)
            cer, wer = cer_wer(TRUTH, r["text"])
            rows.append({"sample": sample, "damage": name, "words": q["words"], "unsure_pct": q["unsure_pct"],
                         "CER_pct": round(100 * cer, 1), "WER_pct": round(100 * wer, 1),
                         "quality": q["level"], "reasons": " | ".join(q["reasons"]),
                         "should_flag": 100 * cer > SHOULD_FLAG_CER, "flagged": q["level"] != "ok"})
            print(f"{sample:6} {name:14} {100 * cer:5.1f}% {100 * wer:5.1f}% {q['unsure_pct']:6.1f}%  "
                  f"{q['level']:7}  {'; '.join(q['reasons'])}")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    avg = lambda key: sum(r[key] for r in rows) / len(rows)
    print(f"\nAverage CER {avg('CER_pct'):.1f}%   average WER {avg('WER_pct'):.1f}%   -> {out_csv}")

    # ---- does the quality check flag the right cases?
    count = lambda f, s: sum(1 for r in rows if r["flagged"] == f and r["should_flag"] == s)
    print(f"\nQuality check vs. 'should be flagged' (CER > {SHOULD_FLAG_CER:g}%):")
    print(f"{'':22}{'should flag':>12}{'should not':>12}")
    print(f"{'flagged (warn/bad)':22}{count(True, True):>12}{count(True, False):>12}")
    print(f"{'not flagged (ok)':22}{count(False, True):>12}{count(False, False):>12}")
    missed = [f"{r['sample']} {r['damage']}" for r in rows if r["should_flag"] and not r["flagged"]]
    extra = [f"{r['sample']} {r['damage']}" for r in rows if r["flagged"] and not r["should_flag"]]
    print("Missed (bad reading, not flagged):", ", ".join(missed) or "none")
    print("False alarms (good reading, flagged):", ", ".join(extra) or "none")

    # ---- pages that must always be flagged
    print("\nExtra checks (must be flagged):")
    photo = load_pages(SAMPLES["photo"])[0]
    extra_cases = [("photo tilted 20 degrees", tilt(photo, 20)),
                   ("blank white page", np.full((2400, 1700, 3), 255, np.uint8))]
    all_ok = True
    for name, img in extra_cases:
        q = reader.page_quality(reader._own_read_page(model, img))
        passed = q["level"] != "ok"
        all_ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:24} {q['level']:5} {'; '.join(q['reasons'])}")
    return rows, all_ok


if __name__ == "__main__":
    run(lexicon="--no-lexicon" not in sys.argv)
