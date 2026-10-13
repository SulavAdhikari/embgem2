"""Benchmark EmbeddingGemma-2 inference speed at 512d. CPU-only, text+vision, no audio."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from infer import OUTPUT_DIM, load_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_image(w: int = 512, h: int = 512, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.text((10, 10), f"synthetic {seed}", fill=(255, 255, 255))
    return img


def timed(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float], n_items: int):
    arr = np.array(times)
    mean = float(arr.mean())
    return {
        "iters": len(times),
        "batch": n_items,
        "mean_s": round(mean, 4),
        "p50_s": round(float(np.median(arr)), 4),
        "p95_s": round(float(np.quantile(arr, 0.95)), 4),
        "per_item_ms": round(mean / n_items * 1000, 2),
        "items_per_s": round(n_items / mean, 3),
    }


def run_case(model, name: str, fn, batch: int, warmup: int, iters: int):
    print(f"[{name}] batch={batch} warmup={warmup} iters={iters} ...", flush=True)
    with torch.no_grad():
        times = timed(fn, warmup, iters)
    s = summarize(times, batch)
    s["case"] = name
    print(f"  mean={s['mean_s']}s p50={s['p50_s']}s p95={s['p95_s']}s | {s['per_item_ms']} ms/item | {s['items_per_s']} it/s")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args()

    print("Loading model (first run downloads ~1.5GB)...")
    t0 = time.perf_counter()
    model = load_model()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s. Dim={OUTPUT_DIM}, device=cpu, dtype=float32")

    short_q = ["What causes the northern lights?"]
    long_q = ["What causes the northern lights? " * 60]  # ~500+ tokens
    img1 = make_image(seed=1)
    img2 = make_image(seed=2)

    results = []
    for b in args.batches:
        texts = (short_q * b)
        results.append(run_case(
            model, "text-short-searchquery",
            lambda texts=texts: model.encode(texts, prompt_name="SearchQuery",
                truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False),
            b, args.warmup, args.iters))

    for b in args.batches:
        texts = (long_q * b)
        results.append(run_case(
            model, "text-long-searchquery",
            lambda texts=texts: model.encode(texts, prompt_name="SearchQuery",
                truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False),
            b, args.warmup, args.iters))

    # Image: encode one at a time (ST vision path); batch loop measures repeated singles
    results.append(run_case(
        model, "image-single-x1",
        lambda: model.encode([img1], truncate_dim=OUTPUT_DIM,
            normalize_embeddings=True, show_progress_bar=False),
        1, args.warmup, args.iters))

    results.append(run_case(
        model, "interleaved-text-plus-1img",
        lambda: model.encode([{"text": "Waterproof running shoes. <|image|>", "image": [img1]}],
            truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False),
        1, args.warmup, args.iters))

    results.append(run_case(
        model, "interleaved-text-plus-2img",
        lambda: model.encode([{"text": "Shoe front <|image|> mesh detail <|image|>", "image": [img1, img2]}],
            truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False),
        1, args.warmup, args.iters))

    # Correctness gate: shape + norm + cross-modal similarity runs
    qe = model.encode(["trail running shoes"], prompt_name="SearchQuery",
        truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False)
    ie = model.encode([{"text": "Running shoes <|image|>", "image": [img1]}],
        truncate_dim=OUTPUT_DIM, normalize_embeddings=True, show_progress_bar=False)
    assert qe.shape == (1, 512) and ie.shape == (1, 512), (qe.shape, ie.shape)
    print(f"sanity shapes {qe.shape} / {ie.shape}, norms {np.linalg.norm(qe):.3f}/{np.linalg.norm(ie):.3f}, "
          f"xmodal sim {float(model.similarity(qe, ie)[0][0]):.3f}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"benchmark_512d_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / f"benchmark_512d_{ts}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "batch", "iters", "mean_s", "p50_s", "p95_s", "per_item_ms", "items_per_s"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote results/benchmark_512d_{ts}.json/csv")


if __name__ == "__main__":
    main()
