# ======================= ROUND 2: train more, on ~20% of the labelled data =======================
# Paste this whole cell at the end of the notebook and run it. Uses everything from cells 1-9.
# Continues from the round-1 best model. Validation and test writers stay the same as round 1,
# so the numbers are directly comparable. Results go to /kaggle/working/results_round2/.
N_TRAIN2         = 400_000   # ~20% of the 2.08 million labelled words
EPOCHS2          = 20
PATIENCE2        = 5
LR2              = 1e-4
CLASS_TEST       = ["ayush", "yash"]   # classmates kept aside for testing (if their data is added)
CLASS_REPEAT     = 10                  # other classmates' words are shown this many times per epoch
OUT2             = "/kaggle/working/results_round2"

import gc, math, time, csv, shutil
os.makedirs(OUT2, exist_ok=True)
with open(f"{OUT2}/config.txt", "w") as f:
    for k in ["N_TRAIN2", "EPOCHS2", "PATIENCE2", "LR2", "CLASS_TEST", "CLASS_REPEAT", "BATCH_SIZE", "SEED"]:
        f.write(f"{k} = {globals()[k]}\n")

# ---------- 1. start from the round-1 best model
cands = [f"{OUT}/htr_weights_best.npz"] + sorted(glob.glob("/kaggle/input/**/htr_weights_best.npz", recursive=True))
START = next((p for p in cands if os.path.exists(p)), None)
if START is None:
    print("WARNING: round-1 model (htr_weights_best.npz) not found - starting from the app weights instead.")
    START = WEIGHTS
model = HTR(len(CHARS) + 1); load_npz(model, START); model = model.to(DEVICE)
print("Starting from:", START)

# ---------- 2. load ~400,000 new training words (never from validation/test writers)
splits["train"] = []; gc.collect()                       # free round-1 training images
held = set(writers["val"]) | set(writers["test"])
train_csvs = [c for c in CSVS if posixpath.dirname(c) not in held]
random.Random(SEED + 1).shuffle(train_csvs)
per_writer = math.ceil(1.2 * N_TRAIN2 / max(1, len(train_csvs)))   # spread over many writers
rng2 = random.Random(SEED + 2)
train2, writers2 = [], []
bar = tqdm(total=N_TRAIN2, desc="round-2 words loaded")
for csv_path in train_csvs:
    if len(train2) >= N_TRAIN2:
        break
    folder = posixpath.dirname(csv_path)
    rows = read_labels(csv_path); rng2.shuffle(rows)
    took = 0
    for fname, text in rows:
        if took >= per_writer or len(train2) >= N_TRAIN2:
            break
        p = resolve(folder, fname)
        img = load_gray(p) if p else None
        if img is None:
            continue
        train2.append((img, text)); took += 1; bar.update(1)
    if took:
        writers2.append(folder)
bar.close()

# ---------- 3. classmates' words (only if training_data.zip / labels.tsv was added as input)
def load_classmates():
    items = []
    for d in [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/labels.tsv", recursive=True)]:
        for line in open(f"{d}/labels.tsv", encoding="utf-8").read().splitlines()[1:]:
            fname, text, student = line.split("\t")
            img = cv2.imread(f"{d}/{fname}", cv2.IMREAD_GRAYSCALE)
            items.append((img, text, student))
    if not items:
        for z in glob.glob("/kaggle/input/**/training_data*.zip", recursive=True):
            with zipfile.ZipFile(z) as cz:
                for line in cz.read("labels.tsv").decode("utf-8").splitlines()[1:]:
                    fname, text, student = line.split("\t")
                    img = cv2.imdecode(np.frombuffer(cz.read(fname), np.uint8), cv2.IMREAD_GRAYSCALE)
                    items.append((img, text, student))
            break
    return [(i, t, s) for i, t, s in items
            if i is not None and t and len(t) <= MAX_LABEL_LEN and all(c in CHARSET for c in t)]
cls = load_classmates()
class_train = [(i, t) for i, t, s in cls if s.lower() not in [x.lower() for x in CLASS_TEST]]
class_test = [(i, t) for i, t, s in cls if s.lower() in [x.lower() for x in CLASS_TEST]]
train_items = train2 + class_train * CLASS_REPEAT

info = [f"Round 2 training words from SSL-HWD-Words: {len(train2):,} from {len(writers2)} writers "
        f"(up to {per_writer} per writer; validation/test writers excluded)",
        f"Classmates' words added to training: {len(class_train):,} x {CLASS_REPEAT} "
        f"(students: {sorted({s for _, _, s in cls if s.lower() not in [x.lower() for x in CLASS_TEST]})})",
        f"Classmates kept aside for testing: {len(class_test):,} words ({CLASS_TEST})" if cls else
        "Classmates' data not added.",
        f"Validation: {len(splits['val']):,} words, test: {len(splits['test']):,} words (same writers as round 1)",
        f"Total training items per epoch: {len(train_items):,}"]
print("\n".join(info)); open(f"{OUT2}/dataset_info.txt", "w").write("\n".join(info) + "\n")

train_loader2 = DataLoader(Words(train_items, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                           collate_fn=collate, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
class_loader = DataLoader(Words(class_test, False), batch_size=BATCH_SIZE, collate_fn=collate) if class_test else None
print("Batches per epoch:", len(train_loader2))

# ---------- 4. scores before round 2 (= after round 1), and of the original app model
orig = HTR(len(CHARS) + 1); load_npz(orig, WEIGHTS); orig = orig.to(DEVICE)
r1_val, r1_test = evaluate(model, val_loader), evaluate(model, test_loader, keep_examples=40)
o_test = evaluate(orig, test_loader)
r1_cls = evaluate(model, class_loader) if class_loader else None
o_cls = evaluate(orig, class_loader) if class_loader else None
print(f"Start of round 2 | val CER {r1_val[1]:.2f}%  words correct {r1_val[2]:.2f}%")

# ---------- 5. train
def write2(name, text, header=None):
    path = f"{OUT2}/{name}"
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new and header: f.write(header + "\n")
        f.write(text + "\n")
for name in os.listdir(OUT2):
    if name.endswith(".txt") and name not in ("config.txt", "dataset_info.txt"):
        os.remove(f"{OUT2}/{name}")
write2("val_loss.txt", f"0\t{r1_val[0]:.5f}", "epoch\tval_loss")
write2("val_cer.txt", f"0\t{r1_val[1]:.3f}", "epoch\tval_cer_percent")
write2("val_word_accuracy.txt", f"0\t{r1_val[2]:.3f}", "epoch\tval_word_accuracy_percent")

opt = torch.optim.Adam(model.parameters(), lr=LR2)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
LAYERS = {f"conv{i+1}": [model.convs[i].weight, model.bns[i].weight, model.bns[i].bias] for i in range(5)}
LAYERS["lstm"] = list(model.lstm.parameters()); LAYERS["output"] = [model.out.weight]
history2, best_cer2, best_epoch2 = [], r1_val[1], 0
save_npz(model, f"{OUT2}/htr_weights_best.npz"); torch.save(model.state_dict(), f"{OUT2}/best_model.pt")

for epoch in range(1, EPOCHS2 + 1):
    t0 = time.time(); model.train()
    loss_sum = n = 0; grad_norms, layer_sums = [], {k: 0.0 for k in LAYERS}
    for x, targets, lengths, texts in tqdm(train_loader2, desc=f"round 2 epoch {epoch}", leave=False):
        logits = model(x.to(DEVICE, non_blocking=True))
        logp = F.log_softmax(logits, 2).permute(1, 0, 2)
        loss = ctc_loss(logp, targets, torch.full((len(texts),), T_STEPS, dtype=torch.long), lengths)
        opt.zero_grad(); loss.backward()
        for k, params in LAYERS.items():
            layer_sums[k] += torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item()
        grad_norms.append(nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        opt.step()
        loss_sum += loss.item() * len(texts); n += len(texts)
    train_loss = loss_sum / n
    val_loss, val_cer, val_acc, _ = evaluate(model, val_loader)
    cls_cer = evaluate(model, class_loader)[1] if class_loader else float("nan")
    lr = opt.param_groups[0]["lr"]; sched.step(val_cer); secs = time.time() - t0
    g_mean, g_max = float(np.mean(grad_norms)), float(np.max(grad_norms))
    per_layer = {k: v / len(train_loader2) for k, v in layer_sums.items()}
    improved = val_cer < best_cer2
    if improved:
        best_cer2, best_epoch2 = val_cer, epoch
        save_npz(model, f"{OUT2}/htr_weights_best.npz"); torch.save(model.state_dict(), f"{OUT2}/best_model.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, f"{OUT2}/last_checkpoint.pt")
    print(f"R2 epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val CER {val_cer:.2f}% "
          f"| words correct {val_acc:.2f}% | classmates CER {cls_cer:.2f}% | grad norm mean {g_mean:.3f} "
          f"max {g_max:.3f} | lr {lr:.1e} | {secs/60:.1f} min{'  <- best, saved' if improved else ''}")
    print("          gradient norm per layer: " + "  ".join(f"{k} {v:.4f}" for k, v in per_layer.items()))
    write2("train_loss.txt", f"{epoch}\t{train_loss:.5f}", "epoch\ttrain_loss")
    write2("val_loss.txt", f"{epoch}\t{val_loss:.5f}")
    write2("val_cer.txt", f"{epoch}\t{val_cer:.3f}")
    write2("val_word_accuracy.txt", f"{epoch}\t{val_acc:.3f}")
    write2("classmates_cer.txt", f"{epoch}\t{cls_cer:.3f}", "epoch\tclassmates_test_cer_percent")
    write2("grad_norm.txt", f"{epoch}\t{g_mean:.5f}\t{g_max:.5f}", "epoch\tmean_grad_norm\tmax_grad_norm")
    write2("grad_norm_per_layer.txt", f"{epoch}\t" + "\t".join(f"{v:.6f}" for v in per_layer.values()),
           "epoch\t" + "\t".join(per_layer))
    write2("learning_rate.txt", f"{epoch}\t{lr:.2e}", "epoch\tlearning_rate")
    write2("epoch_time.txt", f"{epoch}\t{secs:.1f}", "epoch\tseconds")
    history2.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_cer": val_cer,
                     "val_word_accuracy": val_acc, "classmates_cer": cls_cer, "grad_norm_mean": g_mean,
                     "grad_norm_max": g_max, "lr": lr, "seconds": secs,
                     **{f"grad_{k}": v for k, v in per_layer.items()}})
    with open(f"{OUT2}/history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history2[0])); w.writeheader(); w.writerows(history2)
    if epoch - best_epoch2 >= PATIENCE2:
        print(f"No improvement for {PATIENCE2} epochs: stopping."); break

# ---------- 6. graphs
H2 = pd.DataFrame(history2)
def plot2(cols, labels, ylabel, fname, title, base=None):
    plt.figure(figsize=(7, 4.2))
    for c, l in zip(cols, labels):
        plt.plot(H2["epoch"], H2[c], marker="o", label=l)
    if base is not None: plt.axhline(base, ls="--", c="gray", label="start of round 2")
    if best_epoch2: plt.axvline(best_epoch2, ls=":", c="green", label=f"best epoch ({best_epoch2})")
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT2}/{fname}", dpi=150); plt.show()
plot2(["train_loss", "val_loss"], ["training loss", "validation loss"], "CTC loss", "loss_curve.png", "Round 2: training and validation loss")
plot2(["val_cer"], ["validation CER"], "CER (%)", "cer_curve.png", "Round 2: validation CER", r1_val[1])
plot2(["val_word_accuracy"], ["validation word accuracy"], "words correct (%)", "word_accuracy_curve.png", "Round 2: validation word accuracy", r1_val[2])
if class_loader: plot2(["classmates_cer"], ["classmates (held out) CER"], "CER (%)", "classmates_cer_curve.png", "Round 2: classmates' CER", r1_cls[1])
plot2(["grad_norm_mean", "grad_norm_max"], ["mean", "max"], "gradient norm", "grad_norm_curve.png", "Round 2: gradient norm (before clipping at 5)")
gcols = [c for c in H2.columns if c.startswith("grad_") and c not in ("grad_norm_mean", "grad_norm_max")]
plot2(gcols, [c[5:] for c in gcols], "mean gradient norm", "grad_norm_per_layer.png", "Round 2: gradient norm per layer")
plot2(["lr"], ["learning rate"], "learning rate", "learning_rate.png", "Round 2: learning rate")

# ---------- 7. final comparison: original app model -> round 1 -> round 2
model.load_state_dict(torch.load(f"{OUT2}/best_model.pt", map_location=DEVICE))
r2_test = evaluate(model, test_loader, keep_examples=40)
r2_cls = evaluate(model, class_loader) if class_loader else None
row = lambda name, r: f"{name:22}{r[1]:10.2f}{r[2]:18.2f}"
lines = ["SSL-HWD-Words test writers (never trained on)", f"{'model':22}{'CER %':>10}{'words correct %':>18}",
         row("original app model", o_test), row("after round 1", r1_test), row("after round 2", r2_test), ""]
if class_loader:
    lines += [f"Classmates kept aside ({', '.join(CLASS_TEST)}; {len(class_test)} words, never trained on)",
              row("original app model", o_cls), row("after round 1", r1_cls), row("after round 2", r2_cls), ""]
lines.append(f"Round 2 best epoch: {best_epoch2}; validation CER {r1_val[1]:.2f}% -> {best_cer2:.2f}%")
text = "\n".join(lines); print(text); open(f"{OUT2}/test_results.txt", "w").write(text + "\n")
with open(f"{OUT2}/sample_predictions.txt", "w") as f:
    f.write("correct text | after round 1 | after round 2\n")
    for (t, p1), (_, p2) in zip(r1_test[3], r2_test[3]):
        f.write(f"{t} | {p1} | {p2}\n")

# ---------- 8. one file to download
os.chdir("/kaggle/working")
shutil.make_archive("/kaggle/working/results_round2", "zip", OUT2)
from IPython.display import FileLink, display
display(FileLink("results_round2.zip"))
