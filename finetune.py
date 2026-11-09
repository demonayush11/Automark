"""
Fine-tune our handwriting model (CNN + BiLSTM + CTC) on our own students' handwriting.

Run on Google Colab (free GPU) or any computer with PyTorch:

    python finetune.py --data training_data.zip

Inputs (put them next to this file):
    training_data.zip   from the app: Settings -> Training data -> Download
    htr_weights.npz     current model weights (automark/model/htr_weights.npz)
    charList.txt        the 77 characters the model knows (automark/model/charList.txt)
Output:
    htr_weights_finetuned.npz   -> rename to htr_weights.npz and put in automark/model/

How it works
  1. Rebuild the same network as grader/htr_model.py in PyTorch and load the current weights.
  2. Split the words by student: 10% of students are kept aside to test on (never trained on).
  3. Train with a small learning rate, CTC loss, and random image changes (stretch, tilt,
     thicker/thinner strokes) so the model learns the writing style instead of memorising.
  4. After every pass over the data, measure the character error rate (CER) on the test students.
     Keep the best version; stop when it has not improved for a while.
  5. Save the weights back in the exact format the app reads.
"""
import argparse
import io
import os
import random
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_W, IMG_H, T_STEPS, HIDDEN = 128, 32, 32, 256


# ================================================================ data
def preprocess(gray):
    """Exactly the same as grader/htr_model.py preprocess(): fit into 128x32, transpose, normalise."""
    h, w = gray.shape
    f = max(w / IMG_W, h / IMG_H)
    new_w, new_h = max(min(IMG_W, int(w / f)), 1), max(min(IMG_H, int(h / f)), 1)
    target = np.ones((IMG_H, IMG_W)) * 255
    target[:new_h, :new_w] = cv2.resize(gray, (new_w, new_h))
    img = cv2.transpose(target)                    # (128, 32)
    m, s = img.mean(), img.std()
    return ((img - m) / s if s > 0 else img - m).astype(np.float32)


def augment(gray, rng):
    """Random small changes so the model sees many versions of each word."""
    h, w = gray.shape
    # stretch width 80-120%
    gray = cv2.resize(gray, (max(1, int(w * rng.uniform(0.8, 1.2))), h))
    # slight rotation / slant
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), 1)
    m[0, 1] += rng.uniform(-0.2, 0.2)            # shear = slant of the writing
    gray = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    # thicker or thinner strokes
    r = rng.random()
    if r < 0.25:
        gray = cv2.erode(gray, np.ones((2, 2), np.uint8))      # dark ink grows
    elif r < 0.5:
        gray = cv2.dilate(gray, np.ones((2, 2), np.uint8))     # dark ink shrinks
    return gray


def load_samples(path):
    """Read labels.tsv (file<TAB>text<TAB>student) and the word images from a zip or a folder."""
    files = {}
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                files[os.path.basename(name)] = z.read(name)
    else:
        for name in os.listdir(path):
            with open(os.path.join(path, name), "rb") as fh:
                files[name] = fh.read()
    samples = []
    for line in files["labels.tsv"].decode("utf-8").splitlines()[1:]:
        fname, text, student = line.split("\t")
        img = cv2.imdecode(np.frombuffer(files[fname], np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            samples.append((img, text, student))
    return samples


def split_by_student(samples, test_share, seed):
    students = sorted({s for _, _, s in samples})
    random.Random(seed).shuffle(students)
    if len(students) >= 2:
        n_test = max(1, round(len(students) * test_share))
        test_students = set(students[:n_test])
        return ([x for x in samples if x[2] not in test_students],
                [x for x in samples if x[2] in test_students], sorted(test_students))
    shuffled = samples[:]                    # only one student: split words instead
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_share))
    return shuffled[n_test:], shuffled[:n_test], ["(random 10% of words)"]


# ================================================================ model
class HTR(nn.Module):
    """Same network as grader/htr_model.py: 5 x (conv, batch-norm, ReLU, max-pool) -> LSTM read
    forwards and backwards with the SAME weights -> linear layer to 78 classes (77 chars + blank)."""

    def __init__(self, n_classes=78):
        super().__init__()
        chans, kernels = [1, 32, 64, 128, 128, 256], [5, 5, 3, 3, 3]
        self.pools = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        self.convs = nn.ModuleList(nn.Conv2d(chans[i], chans[i + 1], kernels[i], padding=kernels[i] // 2, bias=False)
                                   for i in range(5))
        self.bns = nn.ModuleList(nn.BatchNorm2d(chans[i + 1], eps=1e-3) for i in range(5))
        self.lstm = nn.LSTM(256, HIDDEN, num_layers=2, batch_first=True)
        self.out = nn.Linear(2 * HIDDEN, n_classes, bias=False)

    def forward(self, x):                       # x: (B, 128, 32)
        x = x.unsqueeze(1)                      # (B, 1, 128, 32)
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = F.max_pool2d(F.relu(bn(conv(x))), pool)
        seq = x.squeeze(3).permute(0, 2, 1)     # (B, 32, 256)
        fw, _ = self.lstm(seq)
        bw, _ = self.lstm(torch.flip(seq, [1]))
        return self.out(torch.cat([fw, torch.flip(bw, [1])], dim=2))     # (B, 32, 78)


LSTM_KEY = "bidirectional_rnn__fw__multi_rnn_cell__cell_{}__lstm_cell__{}"


def load_npz(model, path):
    """htr_weights.npz (TensorFlow layout) -> PyTorch model."""
    w = np.load(path)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    with torch.no_grad():
        for i in range(5):
            k = w["Variable" + ("" if i == 0 else f"_{i}")]           # (kh, kw, in, out)
            model.convs[i].weight.copy_(t(k.transpose(3, 2, 0, 1)))  # (out, in, kh, kw)
            bn = "batch_normalization" + ("" if i == 0 else f"_{i}")
            model.bns[i].weight.copy_(t(w[bn + "__gamma"]))
            model.bns[i].bias.copy_(t(w[bn + "__beta"]))
            model.bns[i].running_mean.copy_(t(w[bn + "__moving_mean"]))
            model.bns[i].running_var.copy_(t(w[bn + "__moving_variance"]))
        for layer in range(2):
            kernel = w[LSTM_KEY.format(layer, "kernel")]    # (in + 256, 1024), gates i, j, f, o
            bias = w[LSTM_KEY.format(layer, "bias")]
            n_in = kernel.shape[0] - HIDDEN
            gi, gj, gf, go = np.split(kernel, 4, axis=1)
            bi, bj, bf, bo = np.split(bias, 4)
            k_torch = np.concatenate([gi, gf, gj, go], axis=1)          # PyTorch order: i, f, g, o
            getattr(model.lstm, f"weight_ih_l{layer}").copy_(t(k_torch[:n_in].T))
            getattr(model.lstm, f"weight_hh_l{layer}").copy_(t(k_torch[n_in:].T))
            getattr(model.lstm, f"bias_ih_l{layer}").copy_(t(np.concatenate([bi, bf + 1.0, bj, bo])))  # TF forget_bias=1
            getattr(model.lstm, f"bias_hh_l{layer}").zero_()
        model.out.weight.copy_(t(w["Variable_5"][0, 0].T))


def save_npz(model, path):
    """PyTorch model -> htr_weights.npz in the exact layout the app reads."""
    a = lambda p: p.detach().cpu().numpy().astype(np.float32)
    d = {}
    for i in range(5):
        d["Variable" + ("" if i == 0 else f"_{i}")] = a(model.convs[i].weight).transpose(2, 3, 1, 0)
        bn = "batch_normalization" + ("" if i == 0 else f"_{i}")
        d[bn + "__gamma"], d[bn + "__beta"] = a(model.bns[i].weight), a(model.bns[i].bias)
        d[bn + "__moving_mean"], d[bn + "__moving_variance"] = a(model.bns[i].running_mean), a(model.bns[i].running_var)
    for layer in range(2):
        w_ih, w_hh = a(getattr(model.lstm, f"weight_ih_l{layer}")), a(getattr(model.lstm, f"weight_hh_l{layer}"))
        b = a(getattr(model.lstm, f"bias_ih_l{layer}")) + a(getattr(model.lstm, f"bias_hh_l{layer}"))
        k_torch = np.concatenate([w_ih.T, w_hh.T], axis=0)                 # (in + 256, 1024), order i, f, g, o
        gi, gf, gg, go = np.split(k_torch, 4, axis=1)
        bi, bf, bg, bo = np.split(b, 4)
        d[LSTM_KEY.format(layer, "kernel")] = np.concatenate([gi, gg, gf, go], axis=1)
        d[LSTM_KEY.format(layer, "bias")] = np.concatenate([bi, bg, bf - 1.0, bo])
    d["Variable_5"] = a(model.out.weight).T[None, None]
    np.savez_compressed(path, **d)


# ================================================================ training helpers
def encode(text, idx):
    return [idx[c] for c in text]


def greedy_decode(logits, chars):
    blank = len(chars)
    out = []
    for seq in logits.argmax(2).cpu().numpy():
        txt, prev = [], None
        for k in seq:
            if k != prev and k != blank:
                txt.append(chars[k])
            prev = k
        out.append("".join(txt))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def evaluate(model, data, chars, device, batch=64):
    """Character error rate (%) and word accuracy (%) on (image, text) pairs, no augmentation."""
    model.eval()
    errs = total = right = 0
    with torch.no_grad():
        for i in range(0, len(data), batch):
            chunk = data[i:i + batch]
            x = torch.tensor(np.stack([preprocess(img) for img, _, _ in chunk])).to(device)
            for pred, (_, truth, _) in zip(greedy_decode(model(x), chars), chunk):
                errs += edit_distance(pred, truth)
                total += len(truth)
                right += pred == truth
    return 100 * errs / max(1, total), 100 * right / max(1, len(data))


def set_train_mode(model, freeze_bn):
    model.train()
    if freeze_bn:          # small dataset: keep the batch-norm statistics learnt on IAM
        for bn in model.bns:
            bn.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="training_data.zip", help="zip or folder exported by the app")
    ap.add_argument("--weights", default="htr_weights.npz")
    ap.add_argument("--chars", default="charList.txt")
    ap.add_argument("--out", default="htr_weights_finetuned.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement")
    ap.add_argument("--test-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.chars, encoding="utf-8") as fh:
        chars = fh.read()
    idx = {c: i for i, c in enumerate(chars)}

    samples = [s for s in load_samples(args.data) if s[1] and len(s[1]) <= T_STEPS and all(c in idx for c in s[1])]
    train, test, test_students = split_by_student(samples, args.test_share, args.seed)
    print(f"Device: {device}.  Words: {len(samples)}  (train {len(train)}, test {len(test)})")
    print(f"Test students (never trained on): {', '.join(test_students)}")
    if len(train) < 50:
        print("WARNING: very few training words. Collect more corrected sheets for a real improvement.")

    model = HTR(len(chars) + 1).to(device)
    load_npz(model, args.weights)
    cer, acc = evaluate(model, test, chars, device)
    print(f"Before fine-tuning: test CER {cer:.1f}%   words fully correct {acc:.1f}%")
    best_cer, best_epoch = cer, 0
    save_npz(model, args.out)                     # if nothing improves, the output = the original

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=len(chars), zero_infinity=True)
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, freeze_bn=True)
        random.shuffle(train)
        losses = []
        for i in range(0, len(train), args.batch):
            chunk = train[i:i + args.batch]
            x = torch.tensor(np.stack([preprocess(augment(img, rng)) for img, _, _ in chunk])).to(device)
            targets = [encode(t, idx) for _, t, _ in chunk]
            logp = F.log_softmax(model(x), 2).permute(1, 0, 2)            # (T, B, C) for CTC
            loss = ctc(logp, torch.tensor(sum(targets, []), dtype=torch.long),
                       torch.full((len(chunk),), T_STEPS, dtype=torch.long),
                       torch.tensor([len(t) for t in targets], dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        cer, acc = evaluate(model, test, chars, device)
        mark = ""
        if cer < best_cer:
            best_cer, best_epoch, mark = cer, epoch, "  <- best, saved"
            save_npz(model, args.out)
        print(f"Epoch {epoch:3d}  loss {np.mean(losses):.3f}  test CER {cer:.1f}%  words correct {acc:.1f}%{mark}")
        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs: stopping.")
            break
    print(f"\nBest test CER {best_cer:.1f}% (epoch {best_epoch}).  Saved: {args.out}")
    print("Rename it to htr_weights.npz and copy it into automark/model/ (keep a backup of the old file).")


if __name__ == "__main__":
    main()
