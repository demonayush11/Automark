"""
Dictionary correction for the own handwriting model ("lexicon decoding").

The model outputs, for each of 32 time steps, a probability for every character. Plain (greedy)
decoding takes the single best character at each step, which often gives near-misses like
"crwelty" or "abolutely".

Here, for each word:
  1. Find real English words close to the greedy reading (up to 2 letters different),
     plus two-word splits for joined words ("seebecause" -> "see because").
  2. For each candidate, compute how likely the model's output is for that exact spelling,
     using the CTC forward algorithm (sums over every way the 32 steps could spell the word).
  3. Add a bonus for common words (word frequency), and keep the best-scoring option.
     The greedy reading itself also competes, so a word is only changed when the model
     itself finds the dictionary word about as likely.

Word list: model/english_words.txt (58,954 common English words with frequencies, from wordfreq).
Extra words (e.g. from the marking scheme) can be added so subject terms are recognised.
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS_PATH = os.path.join(os.path.dirname(HERE), "model", "english_words.txt")

FREQ_WEIGHT = 0.7      # bonus per zipf unit of word frequency (zipf: 1 = rare, 7 = "the")
UNKNOWN_ZIPF = 0.0     # frequency given to a reading that is not in the dictionary
SPLIT_PENALTY = 3.0    # cost of turning one word image into two words
EXTRA_ZIPF = 4.5       # frequency given to extra words (marking-scheme vocabulary)
MAX_CANDIDATES = 40

_freq = None           # word -> zipf
_deletes = None        # word with one letter removed -> [words]


def _load():
    global _freq, _deletes
    if _freq is None:
        _freq = {}
        with open(WORDS_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                w, z = line.rstrip("\n").split("\t")
                _freq[w] = float(z)
        _deletes = {}
        for w in _freq:
            _index(w)
    return _freq


def _index(w):
    _deletes.setdefault(w, []).append(w)
    for i in range(len(w)):
        _deletes.setdefault(w[:i] + w[i + 1:], []).append(w)


def add_words(words):
    """Add subject words (e.g. from the marking scheme) so they are not 'corrected' away."""
    freq = _load()
    for w in words:
        w = w.lower()
        if re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", w) and freq.get(w, 0) < EXTRA_ZIPF:
            if w not in freq:
                _index(w)
            freq[w] = EXTRA_ZIPF


def _near(word):
    """Dictionary words within about 2 edits of `word` (delete-index lookup)."""
    keys = {word}
    for i in range(len(word)):
        d1 = word[:i] + word[i + 1:]
        keys.add(d1)
        for j in range(len(d1)):
            keys.add(d1[:j] + d1[j + 1:])
    found = set()
    for k in keys:
        found.update(_deletes.get(k, ()))
    return found


# ---------------------------------------------------------------- CTC scoring
def ctc_logprob(logp, labels, blank):
    """log P(labels | model output) by the CTC forward algorithm. logp: (T, C) log-probabilities."""
    T = logp.shape[0]
    if not labels:
        return float(logp[:, blank].sum())
    ext = [blank]
    for c in labels:
        ext += [c, blank]
    ext = np.array(ext)
    S = len(ext)
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full(S, -np.inf)
    alpha[0] = logp[0, blank]
    alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        a1 = np.concatenate(([-np.inf], alpha[:-1]))
        a2 = np.concatenate(([-np.inf, -np.inf], alpha[:-2]))
        a2[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + logp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def _encode(text, char_index):
    try:
        return [char_index[ch] for ch in text]
    except KeyError:
        return None


def _case_like(word, model_text):
    """Copy the capitalisation pattern of what the model read."""
    if model_text.isupper() and len(model_text) > 1:
        return [word.upper()]
    if model_text[:1].isupper():
        return [word.capitalize()]
    return [word, word.capitalize()] if word != "i" else ["I"]


def correct(greedy, logp, chars):
    """Return the best spelling of one word image. greedy: plain decoding; logp: (T, C) log-probs."""
    freq = _load()
    m = re.match(r"^([^A-Za-z]*)(.*?)([^A-Za-z]*)$", greedy, re.S)
    lead, core, trail = m.group(1), m.group(2), m.group(3)
    if not core or re.search(r"[0-9]", core) or len(core) > 20:
        return greedy                          # numbers, symbols: leave as read
    blank = len(chars)
    char_index = {c: i for i, c in enumerate(chars)}
    low = core.lower()

    def score(text_variants, zipf, penalty=0.0):
        best_txt, best = None, -np.inf
        for txt in text_variants:
            lab = _encode(lead + txt + trail, char_index)
            if lab is None:
                continue
            s = ctc_logprob(logp, lab, blank) + FREQ_WEIGHT * zipf - penalty
            if s > best:
                best_txt, best = txt, s
        return best_txt, best

    # the reading itself competes too (counted as unknown if not a dictionary word)
    best_txt, best = score([core], freq.get(low, UNKNOWN_ZIPF))

    cands = sorted(_near(low), key=lambda w: -freq[w])[:MAX_CANDIDATES]
    for w in cands:
        txt, s = score(_case_like(w, core), freq[w])
        if s > best:
            best_txt, best = txt, s

    # joined words: try every split into two words; one side may be a near-miss ("see"+"becanse")
    if low not in freq and len(low) >= 4:
        for i in range(1, len(low)):
            a, b = low[:i], low[i:]
            a_opts = [a] if a in freq else []
            b_opts = [b] if b in freq else []
            if a_opts and not b_opts and len(b) >= 3:
                b_opts = sorted((w for w in _near(b) if abs(len(w) - len(b)) <= 1), key=lambda w: -freq[w])[:5]
            if b_opts and not a_opts and len(a) >= 3:
                a_opts = sorted((w for w in _near(a) if abs(len(w) - len(a)) <= 1), key=lambda w: -freq[w])[:5]
            for wa in a_opts:
                for wb in b_opts:
                    z = min(freq[wa], freq[wb])
                    for txt in _case_like(wa, core[:i]):
                        lab = _encode(lead + txt + wb + trail, char_index)
                        if lab is None:
                            continue
                        sc = ctc_logprob(logp, lab, blank) + FREQ_WEIGHT * z - SPLIT_PENALTY
                        if sc > best:
                            best_txt, best = txt + " " + wb, sc
    return lead + best_txt + trail
