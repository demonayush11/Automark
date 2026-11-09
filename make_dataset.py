"""
Build fine-tuning data straight from answer-sheet photos and their typed text (no app needed).

Put pairs of files in one folder (or zip):
    aditya_p1.jpg   +  aditya_p1.txt      photo/scan of a page  +  the exact text written on it
    ayush_p1.pdf    +  ayush_p1.txt       (.jpg .jpeg .png .pdf; a PDF should hold one page)

  * The .txt must contain EXACTLY what is written on the page, including the student's spelling
    mistakes, and nothing else (no headings, notes or summaries that are not on the page).
    Keeping the same line breaks as the page gives the best matching; one paragraph also works.
  * A multi-page PDF can have one .txt with a line "Page 2", "Page 3", ... where each page starts;
    the pages are then matched one by one.
  * Words whose label shares few letters with the model's reading are skipped (--min-similarity),
    which limits the damage from text that does not match the page.
  * The part of the file name before the first "_" is the student's name. It is used to keep some
    students aside for testing, so give each student their own prefix.

Run:
    python make_dataset.py --pages pages.zip                -> training_data.zip
    python make_dataset.py --pages my_folder --out data.zip

What it does for every page: cut it into lines and words, read each word with the current model,
line the reading up with the typed text letter by letter, and save each word image with its
correct spelling (same method as the app's "Save as training data" button).
"""
import argparse
import csv
import re
import io
import os
import sys
import tempfile
import zipfile

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grader.reader import get_model                       # noqa: E402
from grader.segment import load_pages                     # noqa: E402
from grader.training_data import label_boxes, read_boxes  # noqa: E402

PAGE_MARK = re.compile(r"^\s*page\s+(\d+)\s*$", re.I)


def split_pages(text, n_pages):
    """Split a text file into pages at lines like "Page 3". Text before the first marker is page 1.
    Returns a list of n_pages texts, or None if the markers do not fit the PDF."""
    pages, current = {}, 1
    for line in text.splitlines():
        m = PAGE_MARK.match(line)
        if m:
            current = int(m.group(1))
            continue
        pages.setdefault(current, []).append(line)
    if n_pages < 2 or len(pages) < 2 or max(pages) != n_pages:
        return None
    return ["\n".join(pages.get(k, [])) for k in range(1, n_pages + 1)]

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".pdf", ".webp", ".bmp", ".tif", ".tiff")


def find_pairs(folder):
    pairs, missing = [], []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXT and not f.startswith("._"):
                txt = os.path.join(root, stem + ".txt")
                if os.path.exists(txt):
                    pairs.append((os.path.join(root, f), txt, stem.split("_")[0].strip() or stem))
                else:
                    missing.append(f)
    return pairs, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True, help="folder or .zip with photo + .txt pairs")
    ap.add_argument("--out", default="training_data.zip")
    ap.add_argument("--min-similarity", type=float, default=0.5,
                    help="skip a word if the model's reading and the label share too few letters "
                         "(0 = keep all; protects against text that is not an exact copy of the page)")
    args = ap.parse_args()

    folder = args.pages
    if folder.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(folder) as z:
            z.extractall(tmp)
        folder = tmp
    pairs, missing = find_pairs(folder)
    if missing:
        print("No .txt found for (skipped):", ", ".join(missing))
    if not pairs:
        sys.exit("No photo + .txt pairs found.")

    model = get_model()
    rows, images = [], {}
    total_skipped = 0
    for n, (img_path, txt_path, student) in enumerate(pairs, 1):
        with open(txt_path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        pages = load_pages(img_path)
        per_page = split_pages(text, len(pages))
        if per_page:          # "Page N" markers match the PDF: align page by page (more accurate)
            labelled, skipped = [], 0
            for page, page_text in zip(pages, per_page):
                boxes, orig = read_boxes([page], model)
                lab, sk = label_boxes(boxes, orig, page_text, model.chars, args.min_similarity)
                labelled += lab
                skipped += sk
        else:
            boxes, orig = read_boxes(pages, model)
            labelled, skipped = label_boxes(boxes, orig, text, model.chars, args.min_similarity)
        total_skipped += skipped
        for k, (img, label) in enumerate(labelled):
            fname = f"p{n:03d}_{k:04d}.png"
            ok, buf = cv2.imencode(".png", img)
            images[fname] = buf.tobytes()
            rows.append({"file": fname, "text": label, "student": student})
        print(f"{os.path.basename(img_path):30} student {student:15} words saved {len(labelled):4}  skipped {skipped}")

    labels = io.StringIO()
    w = csv.DictWriter(labels, fieldnames=["file", "text", "student"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labels.tsv", labels.getvalue())
        for fname, data in images.items():
            z.writestr(fname, data)
    students = len({r["student"] for r in rows})
    print(f"\n{len(rows)} words from {students} student(s) saved to {args.out}  ({total_skipped} skipped)")
    if students < 2:
        print("Tip: use pages from several students so some can be kept aside for testing.")


if __name__ == "__main__":
    main()
