"""
Cut an answer-sheet page into lines, then into words, in reading order.

Steps:
  1. load        PDF pages or photos -> images
  2. clean       grayscale, remove shadows, straighten slight tilt
  3. ink mask    dark pixels = ink; remove ruled lines (long, and short/broken thin pieces) and margins
  4. lines       the page is cut into vertical strips; in each strip the MIDDLE parts of letters show
                 where lines are; pieces are chained across strips, so tilted/curved lines are followed.
                 Line spacing is measured first (repeat distance), so tails of g/y/f don't join lines.
                 Blobs of two touching lines are cut between the lines.
  5. words       inside each line, ink blobs closer than a word-gap are joined into words;
                 "words" that are only specks (dust, paper texture) are dropped
"""
import cv2
import numpy as np

TARGET_WIDTH = 1800
LINE_H_FACTOR = 1.0
GAP_FLOOR = 0.12
LINE_H_MIN = 1.6
TALL_CUT = 2.6


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
def clean_page(bgr):
    """Grayscale page with even white background (removes phone-photo shadows)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = TARGET_WIDTH / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 51)
    norm = cv2.divide(gray, background, scale=255)
    return _deskew(norm)


def _ink_mask(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # very faint pixels are paper texture, not ink
    mask[gray > 215] = 0
    return mask


def _deskew(gray, max_angle=5.0):
    """Try small rotations; keep the one where text rows are sharpest."""
    small = cv2.resize(_ink_mask(gray), None, fx=0.25, fy=0.25)
    best, best_score = 0.0, -1
    h, w = small.shape
    for angle in np.arange(-max_angle, max_angle + 0.01, 0.5):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rot = cv2.warpAffine(small, m, (w, h))
        score = np.var(rot.sum(axis=1))
        if score > best_score:
            best, best_score = angle, score
    if abs(best) < 0.25:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best, 1)
    return cv2.warpAffine(gray, m, (w, h), borderValue=255)


# ---------------------------------------------------------------- 3. ink mask without ruled lines
def ink_without_rules(gray):
    mask = _ink_mask(gray)
    h, w = mask.shape
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 400), 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 3, 450))))
    # shorter pieces of ruled lines (broken under the writing, faint, or curved in a photo):
    # long, very thin horizontal runs. Letters are thicker, so they are not touched.
    short = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 12, 120), 1)))
    thick = cv2.morphologyEx(short, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8))   # parts thicker than 6 px = ink
    short = cv2.subtract(short, cv2.dilate(thick, np.ones((3, 25), np.uint8)))
    horiz = horiz | short
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
    floor = GAP_FLOOR * line_h
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


def _line_pitch(stats, cents, keep, letter_h, height):
    """Distance between written lines, from the repeat distance of the ink-core profile."""
    core = max(2, int(0.3 * letter_h))
    prof = np.zeros(height, float)
    for i in keep:
        cy = int(cents[i][1])
        prof[max(0, cy - core): cy + core + 1] += stats[i][2]
    prof = prof - prof.mean()
    ac = np.correlate(prof, prof, mode="full")[height - 1:]
    lo, hi = int(1.3 * letter_h), min(height - 1, int(6 * letter_h))
    if hi <= lo + 2:
        return 2.5 * letter_h
    seg = ac[lo:hi]
    peaks = [k for k in range(1, len(seg) - 1) if seg[k] >= seg[k - 1] and seg[k] >= seg[k + 1]]
    if not peaks:
        return float(lo + np.argmax(seg))
    top = max(seg[k] for k in peaks)
    first = next(k for k in peaks if seg[k] >= 0.5 * top)     # first clear repeat, not a multiple of it
    return float(lo + first)


def _bands_1d(prof, letter_h, pitch):
    """Row ranges where the profile has ink cores; too-tall ranges split at their emptiest rows;
    pieces closer than about half a line are one line."""
    if prof.max() == 0:
        return []
    on = prof > 0.15 * np.percentile(prof[prof > 0], 90)
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y]); start = None
    if start is not None:
        bands.append([start, len(prof)])
    out = []
    for a, b in bands:
        stack = [(a, b)]
        while stack:
            a2, b2 = stack.pop()
            if b2 - a2 > 1.3 * pitch:
                lo, hi = a2 + int(0.4 * pitch), b2 - int(0.4 * pitch)
                cut = lo + int(np.argmin(prof[lo:hi]))
                stack += [(cut, b2), (a2, cut)]
            else:
                out.append((a2, b2))
    out.sort()
    merged = []
    for a, b in out:
        if merged and (a + b) / 2 - (merged[-1][0] + merged[-1][1]) / 2 < 0.55 * pitch:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _find_text_lines(stats, cents, keep, letter_h, height, width, pitch):
    """Find text lines that may be tilted or curved (phone photos, sloping writing).
    The page is cut into vertical strips; in each strip, rows holding the MIDDLE part of letters
    form short line pieces; pieces in neighbouring strips at about the same height are chained.
    Returns a list of lines, each a function-like pair (xs, ys) of its centre height across the page."""
    core = max(2, int(0.3 * letter_h))
    n_strips = int(np.clip(width / (7 * letter_h), 3, 10))
    edges = np.linspace(0, width, n_strips + 1)
    xs_mid = (edges[:-1] + edges[1:]) / 2
    pieces = []                                   # per strip: list of centre heights
    for s in range(n_strips):
        prof = np.zeros(height, float)
        for i in keep:
            if edges[s] <= cents[i][0] < edges[s + 1]:
                cy = int(cents[i][1])
                prof[max(0, cy - core): cy + core + 1] += stats[i][2]
        prof = np.convolve(prof, np.ones(5) / 5, mode="same")
        pieces.append([(a + b) / 2 for a, b in _bands_1d(prof, letter_h, pitch)])
    lines = []                                    # each: {"x": [...], "y": [...]}
    for s in range(n_strips):
        taken = set()
        cands = []
        for c in pieces[s]:
            for k, ln in enumerate(lines):
                gap = s - ln["s"][-1]
                if gap < 1 or gap > 3:
                    continue
                slope = ln["slope"]
                pred = ln["y"][-1] + slope * (xs_mid[s] - ln["x"][-1])
                d = abs(c - pred)
                if d < 0.35 * pitch:
                    cands.append((d, c, k))
        cands.sort()
        used_c = set()
        for d, c, k in cands:
            if c in used_c or k in taken:
                continue
            ln = lines[k]
            new_slope = (c - ln["y"][-1]) / (xs_mid[s] - ln["x"][-1])
            ln["slope"] = 0.5 * ln["slope"] + 0.5 * new_slope if len(ln["x"]) > 1 else new_slope
            ln["x"].append(xs_mid[s]); ln["y"].append(c); ln["s"].append(s)
            taken.add(k); used_c.add(c)
        for c in pieces[s]:
            if c not in used_c:
                lines.append({"x": [xs_mid[s]], "y": [c], "s": [s], "slope": 0.0})
    # join pieces of the same line that were not chained (blank stretch, big gap between words)
    changed = True
    while changed:
        changed = False
        best = None
        for a, A in enumerate(lines):
            for b, B in enumerate(lines):
                if a == b or A["s"][-1] >= B["s"][0]:
                    continue
                d = max(abs(_line_y(A, B["x"][0]) - B["y"][0]), abs(_line_y(B, A["x"][-1]) - A["y"][-1]))
                if d < 0.3 * pitch and (best is None or d < best[0]):
                    best = (d, a, b)
        if best:
            _, a, b = best
            A, B = lines[a], lines[b]
            A["x"] += B["x"]; A["y"] += B["y"]; A["s"] += B["s"]
            del lines[b]
            changed = True
    return lines


def _line_y(ln, x):
    """Centre height of a line at position x (straight continuation beyond its ends)."""
    xs, ys = ln["x"], ln["y"]
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
    return float(np.interp(x, xs, ys))


def segment_page(bgr):
    """Full pipeline for one page.
    Returns (gray_page, lines) where lines = [[{"box": (x0,y0,x1,y1), "img": crop}, ...], ...]."""
    gray = clean_page(bgr)
    mask, rules = ink_without_rules(gray)
    clean = gray.copy()
    clean[rules > 0] = 255                     # erase ruled lines from what the model sees

    mask = remove_non_text(mask)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return gray, []
    letter_h = _letter_height(stats)
    min_area = max(12, int(0.005 * letter_h * letter_h))
    H, W = mask.shape

    def usable(i):
        x, y, w, h, area = stats[i]
        if area < min_area:
            return False
        if h < 0.35 * letter_h and w > 1.5 * letter_h and w > 5 * h:   # thin stroke left from a ruled line
            return False
        if w < 0.25 * letter_h and h > 2.2 * letter_h:     # piece of a vertical margin line
            return False
        return True

    keep = [i for i in range(1, n) if usable(i) and stats[i][3] < 3 * letter_h]
    pitch = _line_pitch(stats, cents, keep, letter_h, H)
    lines = _find_text_lines(stats, cents, keep, letter_h, H, W, pitch)
    if not lines:
        return gray, []

    def nearest_line(cx, cy):
        d = [abs(cy - _line_y(ln, cx)) for ln in lines]
        j = int(np.argmin(d))
        return j, d[j]

    # a blob taller than ~2 lines (letters of two lines touching): cut it between the lines
    tall = [i for i in range(1, n) if usable(i) and stats[i][3] >= TALL_CUT * letter_h]
    for i in tall:
        x, y, w, h, _ = stats[i]
        cx = x + w / 2
        ys = sorted(_line_y(ln, cx) for ln in lines)
        for a, b in zip(ys, ys[1:]):
            c = int((a + b) / 2)
            if b - a > 0.5 * pitch and y + 0.25 * letter_h < c < y + h - 0.25 * letter_h:
                sub = labels[c - 1:c + 2, x:x + w] == i
                mask[c - 1:c + 2, x:x + w][sub] = 0
    if tall:
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

    # give every blob to the line passing nearest to the blob's centre.
    big_area = max(12, int(0.02 * letter_h * letter_h))
    per_line = [[] for _ in lines]
    small_line = [[] for _ in lines]
    for i in range(1, n):
        if not usable(i) or stats[i][3] > 4 * letter_h:
            continue
        x, y, w, h, area = stats[i]
        j, d = nearest_line(cents[i][0], cents[i][1])
        if d <= 0.8 * pitch:
            per_line[j].append((x, y, x + w, y + h, {i}))
    # reading order: top to bottom, by each line's height at the middle of the page
    order = sorted(range(len(lines)), key=lambda k: _line_y(lines[k], W / 2))
    per_line = [per_line[k] for k in order]
    small_line = [small_line[k] for k in order]
    centres = sorted(_line_y(ln, W / 2) for ln in lines)

    # height of a written line, for the word-gap rule: from the distance between line centres
    out = []
    for j, comps in enumerate(per_line):
        if not comps:
            continue
        ink = sum(int(stats[i][4]) for c in comps for i in c[4])
        if ink < 0.15 * letter_h * letter_h and max(c[3] - c[1] for c in comps) < 0.5 * letter_h:
            continue                     # only dots/specks on this "line" (i-dots, dust): not text
        # height of this line's writing (tall letters), for the word-gap rule
        line_h = max(LINE_H_FACTOR * float(np.percentile([c[3] - c[1] for c in comps], 80)), LINE_H_MIN * letter_h)
        # a "word" made only of specks (paper texture, dust, a stray dot) is not a word
        groups = [g for g in split_words(comps, line_h)
                  if sum(int(stats[i][4]) for i in g[4]) >= big_area]
        words = []
        for x0, y0, x1, y1, ids in groups:
            p = 4
            X0, X1 = max(0, x0 - p), min(W, x1 + p)
            Y0, Y1 = max(0, y0 - p), min(H, y1 + p)
            crop = clean[Y0:Y1, X0:X1].copy()
            # keep only this word's ink (+ a thin halo); blank out paper and other lines' strokes
            mine = np.isin(labels[Y0:Y1, X0:X1], list(ids)).astype(np.uint8)
            mine = cv2.dilate(mine, np.ones((7, 7), np.uint8))
            crop[mine == 0] = 255
            words.append({"box": (int(X0), int(Y0), int(X1), int(Y1)), "img": crop})
        out.append(words)
    return gray, out


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
