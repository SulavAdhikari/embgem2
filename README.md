# EmbeddingGemma-2 — text+vision inference, 512d, CPU-only (no CUDA, no audio)

Model: https://huggingface.co/google/embeddinggemma-2 — 740M total, loaded as ~440M text+vision via `audio_config=None`.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\cabba\OneDrive\Desktop\ResearchMaterials\embgem2
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` pins CPU `torch` (extra-index `https://download.pytorch.org/whl/cpu`) + `sentence-transformers transformers accelerate Pillow numpy tqdm`.

Rules from model card, enforced in code:
- `config_kwargs={"audio_config": None}` — drops 300M audio encoder. Never pass `audio`/`video` inputs.
- `truncate_dim=512, normalize_embeddings=True` — MRL slice + mandatory L2 renorm; query and corpus must share 512d.
- `torch.float32` on CPU — NEVER `float16` (NaN / silent degradation). `bfloat16` only on CUDA.
- Text task prefixes: `SearchQuery` for queries, `Document` (`title: none | text: ...`) for corpus, `Classification`/`Clustering`/`SentenceSimilarity` symmetric. Images take no prefix; interleaved text uses `<|image|>` placeholders in order.

## Run

```powershell
python infer.py          # smoke: loads model, prints 512d shapes + similarity
python benchmark.py --warmup 3 --iters 20 --batches 1 8
```

Results go to `results/benchmark_512d_<ts>.json/csv` with mean/p50/p95, ms/item, items/sec for: text-short, text-long, image-single, interleaved-1img, interleaved-2img.

Expect CPU (Intel Iris Xe host, torch CPU): ~0.5–2 s per short text single, higher for images/interleaved. First run downloads ~1.5 GB.
