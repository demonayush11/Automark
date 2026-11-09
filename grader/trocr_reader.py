"""
Second reader: TrOCR (Microsoft), a handwriting OCR model that runs on this computer.

No API is called. The model files (about 1.3 GB) are downloaded once from Hugging Face
on first use, then everything runs offline.

TrOCR is an encoder-decoder Transformer:
  * encoder: a Vision Transformer (ViT) cuts the line image into 16x16 patches and turns them into features
  * decoder: a text Transformer writes the text one token at a time, looking at those features
It reads a whole text LINE at a time. The lines come from segment.py (same cutting as the own model).

Model: microsoft/trocr-base-handwritten (fine-tuned on the IAM handwriting dataset).
Change with the TROCR_MODEL environment variable, e.g. microsoft/trocr-small-handwritten (smaller, faster).
"""
import os

import numpy as np

TROCR_MODEL = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
LOW_CONF = 0.5        # a line whose confidence is below this gets [?] at the end

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        try:
            import torch  # noqa: F401
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise RuntimeError("TrOCR needs torch and transformers. Run: pip install -r requirements.txt") from e
        try:
            _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
            _model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
        except OSError as e:
            raise RuntimeError(f"Could not load TrOCR model '{TROCR_MODEL}'. The first use needs internet "
                               f"to download it (about 1.3 GB). Details: {e}") from e
        _model.eval()
    return _processor, _model


def line_crops(gray, lines, pad=10):
    """One image per text line: the area covering all word boxes of that line."""
    crops = []
    h, w = gray.shape
    for words in lines:
        x0 = max(0, min(wd["box"][0] for wd in words) - pad)
        y0 = max(0, min(wd["box"][1] for wd in words) - pad)
        x1 = min(w, max(wd["box"][2] for wd in words) + pad)
        y1 = min(h, max(wd["box"][3] for wd in words) + pad)
        crops.append(gray[y0:y1, x0:x1])
    return crops


def read_lines(line_images, batch_size=8):
    """line_images: list of grayscale line crops. Returns list of (text, confidence 0..1)."""
    import torch
    from PIL import Image
    processor, model = _load()
    out = []
    for start in range(0, len(line_images), batch_size):
        chunk = [Image.fromarray(np.asarray(im, dtype=np.uint8)).convert("RGB")
                 for im in line_images[start:start + batch_size]]
        pixels = processor(images=chunk, return_tensors="pt").pixel_values
        with torch.no_grad():
            gen = model.generate(pixels, max_new_tokens=64, num_beams=1,
                                 return_dict_in_generate=True, output_scores=True)
        texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
        # confidence = average probability of the tokens the model chose
        scores = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        pad_id = model.generation_config.pad_token_id
        chosen = gen.sequences[:, 1:]
        for i, text in enumerate(texts):
            keep = chosen[i] != pad_id if pad_id is not None else torch.ones_like(chosen[i], dtype=torch.bool)
            logp = scores[i][keep[: scores.shape[1]]]
            conf = float(torch.exp(logp.mean())) if logp.numel() else 0.0
            out.append((text.strip(), conf))
    return out
