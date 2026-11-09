"""
Handwriting reader (HTR = Handwritten Text Recognition).

This is the CNN + RNN + CTC model from the AutoEx project (github.com/4bdul4ziz/AutoEx),
trained on the IAM handwriting dataset (saved version "snapshot-20", the most accurate of the
three versions in that repo on our tests). Its architecture comes from SimpleHTR
(github.com/githubharald/SimpleHTR).

The original code needs an old TensorFlow 1.x graph. Here the same trained weights
are run with plain NumPy, so no TensorFlow install is needed. The maths is identical:

    word image (128 x 32)
      -> 5 x [convolution -> batch-norm -> ReLU -> max-pool]   (CNN: finds pen-stroke features)
      -> 2-layer bidirectional LSTM                              (RNN: reads features left->right and right->left)
      -> 1x1 projection to 78 scores per time step               (77 characters + 1 "blank")
      -> CTC greedy decoding                                     (merge repeats, drop blanks -> text)
      -> dictionary correction (lexicon.py)                      (pick the most likely real English word)

The model reads ONE WORD image at a time. Cutting a page into words is done in segment.py.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "model")

IMG_W, IMG_H = 128, 32   # model input size (width, height)
MAX_T = 32               # time steps the RNN outputs
BN_EPS = 1e-3            # TF1 batch_normalization default epsilon
HIDDEN = 256
USE_LEXICON = True       # dictionary correction after decoding (see lexicon.py)


# ---------------------------------------------------------------- preprocessing
def _ricker_kernel(size=5, sigma=1.0):
    x = np.arange(size) - (size - 1) / 2
    k = (1 - x**2 / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def preprocess(gray, use_ricker=False):
    """Same steps as AutoEx SamplePreprocessor.preprocess (inference mode).
    gray: 2-D uint8 image of ONE word, dark ink on light paper."""
    if gray is None or gray.size == 0:
        gray = np.full((IMG_H, IMG_W), 255, np.uint8)
    h, w = gray.shape
    f = max(w / IMG_W, h / IMG_H)
    new_w = max(min(IMG_W, int(w / f)), 1)
    new_h = max(min(IMG_H, int(h / f)), 1)
    small = cv2.resize(gray, (new_w, new_h))
    target = np.ones((IMG_H, IMG_W)) * 255
    target[:new_h, :new_w] = small
    img = cv2.transpose(target)                          # shape (128, 32)
    if use_ricker:
        img = cv2.filter2D(img, -1, _ricker_kernel())
    m, s = img.mean(), img.std()
    img = img - m
    return img / s if s > 0 else img


# ---------------------------------------------------------------- network layers
def _conv_same(x, k):
    """x: (B,H,W,C)  k: (kh,kw,C,O)  -> (B,H,W,O). Stride 1, SAME padding (TF semantics)."""
    kh, kw = k.shape[:2]
    pt, pl = (kh - 1) // 2, (kw - 1) // 2
    xp = np.pad(x, ((0, 0), (pt, kh - 1 - pt), (pl, kw - 1 - pl), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))  # (B,H,W,C,kh,kw)
    return np.tensordot(win, k, axes=([4, 5, 3], [0, 1, 2]))


def _maxpool(x, ph, pw):
    b, h, w, c = x.shape
    x = x[:, : h // ph * ph, : w // pw * pw, :]
    return x.reshape(b, h // ph, ph, w // pw, pw, c).max(axis=(2, 4))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lstm_layer(seq, kernel, bias):
    """TF1 LSTMCell (gate order i, j, f, o; forget_bias=1). seq: (B,T,F) -> (B,T,H)."""
    b, t, _ = seq.shape
    h = np.zeros((b, HIDDEN), np.float32)
    c = np.zeros((b, HIDDEN), np.float32)
    out = np.empty((b, t, HIDDEN), np.float32)
    for step in range(t):
        z = np.concatenate([seq[:, step], h], axis=1) @ kernel + bias
        i, j, f, o = np.split(z, 4, axis=1)
        c = _sigmoid(f + 1.0) * c + _sigmoid(i) * np.tanh(j)
        h = _sigmoid(o) * np.tanh(c)
        out[:, step] = h
    return out


class HandwritingModel:
    def __init__(self, model_dir=MODEL_DIR, use_ricker=False):
        w = np.load(os.path.join(model_dir, "htr_weights.npz"))
        self.w = {k.replace("__", "/"): w[k].astype(np.float32) for k in w.files}
        with open(os.path.join(model_dir, "charList.txt"), encoding="utf-8") as fh:
            self.chars = fh.read()
        self.blank = len(self.chars)
        self.use_ricker = use_ricker

    def _logits(self, imgs):
        x = np.asarray(imgs, np.float32)[..., None]                  # (B,128,32,1)
        pools = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        for i, (ph, pw) in enumerate(pools):
            bn = "batch_normalization" + ("" if i == 0 else f"_{i}")
            k = self.w["Variable" + ("" if i == 0 else f"_{i}")]
            x = _conv_same(x, k)
            x = (x - self.w[bn + "/moving_mean"]) / np.sqrt(self.w[bn + "/moving_variance"] + BN_EPS)
            x = x * self.w[bn + "/gamma"] + self.w[bn + "/beta"]
            x = _maxpool(np.maximum(x, 0), ph, pw)
        seq = x[:, :, 0, :]                                           # (B,32,256)

        def stack(s):
            for layer in range(2):
                p = f"bidirectional_rnn/fw/multi_rnn_cell/cell_{layer}/lstm_cell/"
                s = _lstm_layer(s, self.w[p + "kernel"], self.w[p + "bias"])
            return s

        fw = stack(seq)
        bw = stack(seq[:, ::-1])[:, ::-1]          # same weights read right-to-left
        both = np.concatenate([fw, bw], axis=2)     # (B,32,512)
        return both @ self.w["Variable_5"][0, 0]    # (B,32,78)

    def read_words(self, word_images, batch_size=32):
        """word_images: list of grayscale word crops.
        Returns list of (text, confidence 0..1)."""
        results = []
        for start in range(0, len(word_images), batch_size):
            chunk = word_images[start:start + batch_size]
            logits = self._logits([preprocess(im, self.use_ricker) for im in chunk])
            e = np.exp(logits - logits.max(axis=2, keepdims=True))
            probs = e / e.sum(axis=2, keepdims=True)
            for p in probs:
                best = p.argmax(axis=1)
                text, char_p, prev = [], [], None
                for t, idx in enumerate(best):
                    if idx != prev and idx != self.blank:
                        text.append(self.chars[idx])
                        char_p.append(p[t, idx])
                    prev = idx
                # confidence = lowest probability among the characters the model wrote
                conf = float(min(char_p)) if char_p else 0.0
                word = "".join(text)
                if USE_LEXICON and word:
                    from . import lexicon
                    word = lexicon.correct(word, np.log(p + 1e-12), self.chars)
                results.append((word, conf))
        return results
