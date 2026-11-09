"""
Cut an answer-sheet page into lines, then into words, in reading order.

Steps:
  1. load        PDF pages or photos -> images
  2. clean       grayscale, remove shadows, straighten slight tilt
  3. ink mask    dark pixels = ink; remove long ruled lines (notebook lines, margins)
  4. lines       rows with ink are grouped into text lines (horizontal projection)
  5. words       inside each line, ink blobs closer than a word-gap are joined into words
"""
import cv2
import numpy as np

TARGET_WIDTH = 1800  # pages are resized to this width so thresholds behave the same


# ---------------------------------------------------------------- 1. load
def load_pages(path):
    """Return list of BGR images from a PDF, or a single photo/scan."""
    if path.lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        pages = []
        for i in range(len(pdf)):
            pil = pdf[i].render(scale=200 / 72).to_pil().convert("RGB")
            pages.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        return pages
    data = np.fromfile(path, np.uint8)          # works with non-ASCII paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot open image: {path}")
    return [img]


# ---------------------------------------------------------------- 2. clean
def noise_level(gray):
    """Estimated grain/noise strength: spread of (image - its 3x3 median), robust to edges."""
    d = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return 1.4826 * float(np.median(np.abs(d - np.median(d))))


def clean_page(bgr):
    """Grayscale page with even white background (removes phone-photo shadows and grain).
    Returns (gray_page, tilt_angle_in_degrees_that_was_corrected)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = TARGET_WIDTH / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    sigma = noise_level(gray)
    if sigma > 2.5:   # grainy photo (dark room, cheap camera): non-local-means denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=1.5 * sigma, templateWindowSize=7, searchWindowSize=21)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 51)
    norm = cv2.divide(gray, background, scale=255)
    return _deskew(norm)


def _ink_mask(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # very faint pixels are paper texture, not ink
    mask[gray > 215] = 0
    return mask


def _deskew(gray, max_angle=15.0):
    """Straighten a tilted page: try rotations, keep the one where text rows are sharpest
    (rows of the page alternate clearly between full of ink and empty).
    Coarse search every 1 degree up to +-15, then fine search every 0.1 degree around the best.
    Returns (straightened_page, angle). An angle near the +-15 limit means the page may be tilted more."""
    small = cv2.resize(_ink_mask(gray), None, fx=0.25, fy=0.25)
    h, w = small.shape
    if not small.any():                       # no ink at all: nothing to straighten
        return gray, 0.0

    def score(angle):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return np.var(cv2.warpAffine(small, m, (w, h)).sum(axis=1))

    best = max(np.arange(-max_angle, max_angle + 0.01, 1.0), key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    best = max(fine[np.abs(fine) <= max_angle + 1e-9], key=score)     # stay within the search limit
    best = float(round(best, 1))
    if abs(best) < 0.1:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1)
    return cv2.warpAffine(gray, m, (w, h), borderValue=255), best


# ---------------------------------------------------------------- 3. ink mask without ruled lines
def ink_without_rules(gray):
    mask = _ink_mask(gray)
    h, w = mask.shape
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 400), 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 3, 450))))
    rules = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    no_rules = cv2.subtract(mask, rules)
    # letters that cross a ruled line got cut in two: re-join them vertically, only inside the rule zone
    zone = cv2.dilate(horiz, np.ones((5, 1), np.uint8))
    rejoined = cv2.morphologyEx(no_rules, cv2.MORPH_CLOSE, np.ones((11, 1), np.uint8))
    mask = no_rules | (rejoined & zone)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop dust
    rules = cv2.subtract(rules, mask)          # what is really only ruled line
    return mask, rules


def _letter_height(stats):
    """Typical height of a letter/word blob. Uses the blobs that hold most of the ink,
    so paper specks and dots don't drag it down."""
    if len(stats) <= 1:
        return 20.0
    areas, heights = stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_HEIGHT]
    order = np.argsort(areas)[::-1]
    covered = np.cumsum(areas[order]) / max(areas.sum(), 1)
    main = order[: max(1, int(np.searchsorted(covered, 0.8)) + 1)]
    return float(np.median(heights[main]))


def remove_non_text(mask):
    """Drop blobs that are not handwriting:
    - binder / punch holes and big ink stains: filled, roughly round or square, solid
    - marks touching the left/right edge of the page (holes cut off, page border, shadow)"""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    letter_h = _letter_height(stats)
    drop = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        fill = area / float(bw * bh)
        roundish = 0.6 < bw / max(bh, 1) < 1.6
        if fill > 0.6 and roundish and min(bw, bh) > 0.6 * letter_h:      # solid disc / block
            drop[i] = True
        elif (x <= 2 or x + bw >= w - 2) and bh > 3 * letter_h:             # tall mark on the page edge
            drop[i] = True
        elif bw > 0.6 * w and bh > 0.5 * h:                                 # page border / big frame
            drop[i] = True
    if not drop.any():
        return mask
    return np.where(drop[labels], 0, mask).astype(np.uint8)


# ---------------------------------------------------------------- 4. lines
def find_lines(mask, letter_h=None):
    """Return list of (top, bottom) row ranges that contain a text line."""
    profile = mask.sum(axis=1).astype(float) / 255
    profile = np.convolve(profile, np.ones(9) / 9, mode="same")
    if profile.max() == 0:
        return []
    on = profile > max(2.0, 0.08 * np.percentile(profile[profile > 0], 90))
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y]); start = None
    if start is not None:
        bands.append([start, len(on)])
    if not bands:
        return []
    heights = np.array([b - a for a, b in bands])
    typical = np.median(heights)
    if letter_h:                       # a text line is at least about one letter tall
        typical = max(typical, letter_h)
    # 1) merge a thin band (dots, accents, descender tails) into its neighbour
    merged = []
    for a, b in bands:
        if merged and (b - a < 0.35 * typical or a - merged[-1][1] < 0.15 * typical):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    # 2) two lines touching (long tails of g, y, p): split a too-tall band at its emptiest row
    split = []
    for a, b in merged:
        stack = [(a, b)]
        while stack:
            a2, b2 = stack.pop()
            if b2 - a2 > 1.6 * typical and (not letter_h or b2 - a2 > 2.8 * letter_h):
                lo, hi = a2 + int(0.3 * typical), b2 - int(0.3 * typical)
                cut = lo + int(np.argmin(profile[lo:hi]))
                stack += [(cut, b2), (a2, cut)]
            else:
                split.append([a2, b2])
    return [(a, b) for a, b in sorted(split) if b - a >= 0.3 * typical]


# ---------------------------------------------------------------- 5. words
def _gap_threshold(gaps, line_h):
    """Decide which blank gaps separate words (big) and which separate letters (small).
    1-D Otsu split of the gap widths, with a sensible minimum."""
    floor = 0.12 * line_h
    if len(gaps) == 0:
        return np.inf
    # cap very wide gaps so one far-away blob can't pull the threshold up
    gaps = np.sort(np.minimum(np.asarray(gaps, float), 0.8 * line_h))
    if len(gaps) == 1 or gaps[-1] < 1.8 * gaps[0]:
        # gaps all similar: either all word gaps or all letter gaps
        return gaps[0] if gaps[0] > 0.3 * line_h else max(gaps[-1] + 1, floor)
    best_t, best_var = gaps[-1], -1
    for i in range(1, len(gaps)):
        a, b = gaps[:i], gaps[i:]
        var = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, (gaps[i - 1] + gaps[i]) / 2
    return min(max(best_t, floor), 0.6 * line_h)   # a gap of 0.6 line-heights is always a word gap


def split_words(comps, line_h):
    """comps: list of component boxes (x0,y0,x1,y1) of one line. Returns grouped word boxes."""
    comps = sorted(comps, key=lambda c: c[0])
    # merge overlapping columns into ink runs, then look at blank gaps between runs
    runs = []
    for c in comps:
        if runs and c[0] <= runs[-1][2]:
            r = runs[-1]
            runs[-1] = (r[0], min(r[1], c[1]), max(r[2], c[2]), max(r[3], c[3]), r[4] | c[4])
        else:
            runs.append(c)
    gaps = [runs[i + 1][0] - runs[i][2] for i in range(len(runs) - 1)]
    t = _gap_threshold(gaps, line_h)
    words = [list(runs[0])] if runs else []
    for i, r in enumerate(runs[1:]):
        if gaps[i] >= t:
            words.append(list(r))
        else:
            w = words[-1]
            words[-1] = [w[0], min(w[1], r[1]), max(w[2], r[2]), max(w[3], r[3]), w[4] | r[4]]
    return [tuple(w) for w in words]


def segment_page(bgr):
    """Full pipeline for one page.
    Returns (gray_page, lines, info):
      lines = [[{"box": (x0,y0,x1,y1), "img": crop}, ...], ...]   (one list of words per text line)
      info  = {"angle": tilt in degrees that was corrected}"""
    gray, angle = clean_page(bgr)
    info = {"angle": angle}
    mask, rules = ink_without_rules(gray)
    clean = gray.copy()
    clean[rules > 0] = 255                     # erase ruled lines from what the model sees

    mask = remove_non_text(mask)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    letter_h = _letter_height(stats) if n > 1 else None
    bands = find_lines(mask, letter_h)
    if not bands:
        return gray, [], info
    typical = float(np.median([b - a for a, b in bands]))
    min_area = max(12, int(0.004 * typical * typical))

    # give every ink blob (letter / joined letters) to the text line its centre falls in
    per_line = [[] for _ in bands]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cy = cents[i][1]
        dist = [0 if a <= cy <= b else min(abs(cy - a), abs(cy - b)) for a, b in bands]
        j = int(np.argmin(dist))
        if dist[j] <= 0.6 * typical:
            per_line[j].append((x, y, x + w, y + h, {i}))

    out = []
    for j, comps in enumerate(per_line):
        if not comps:
            continue
        ink = sum(int(stats[i][4]) for c in comps for i in c[4])
        if letter_h and ink < 0.15 * letter_h * letter_h and max(c[3] - c[1] for c in comps) < 0.5 * letter_h:
            continue                     # only dots/specks on this "line" (i-dots, dust): not text
        line_h = bands[j][1] - bands[j][0]
        words = []
        for x0, y0, x1, y1, ids in split_words(comps, line_h):
            p = 4
            X0, X1 = max(0, x0 - p), min(gray.shape[1], x1 + p)
            Y0, Y1 = max(0, y0 - p), min(gray.shape[0], y1 + p)
            crop = clean[Y0:Y1, X0:X1].copy()
            # keep only this word's ink (+ a thin halo); blank out paper and other lines' strokes
            mine = np.isin(labels[Y0:Y1, X0:X1], list(ids)).astype(np.uint8)
            mine = cv2.dilate(mine, np.ones((7, 7), np.uint8))
            crop[mine == 0] = 255
            words.append({"box": (int(X0), int(Y0), int(X1), int(Y1)), "img": crop})
        out.append(words)
    return gray, out, info


def draw_boxes(gray, lines, texts=None):
    """Debug/preview picture: green box per word, line number on the left."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    k = 0
    for li, words in enumerate(lines):
        for w in words:
            x0, y0, x1, y1 = w["box"]
            conf = texts[k][1] if texts else 1.0
            color = (40, 160, 40) if conf >= 0.5 else (30, 120, 230)   # green = sure, orange = check
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            k += 1
        x0, y0 = words[0]["box"][:2]
        cv2.putText(vis, str(li + 1), (max(0, x0 - 40), y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 50, 50), 2)
    return vis
