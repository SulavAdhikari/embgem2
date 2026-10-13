"""EmbeddingGemma-2 inference: text + visual only, 512d output. CPU-only, no CUDA."""
from __future__ import annotations

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
OUTPUT_DIM = 512  # MRL truncation: 128/256/512/768 supported, we fix 512

# Task prefixes from model card. Text-only; images never take prefixes.
PROMPTS = {
    "SearchQuery": "task: search result | query: {q}",
    "QuestionAnswering": "task: question answering | query: {q}",
    "FactChecking": "task: fact checking | query: {q}",
    "CodeRetrieval": "task: code retrieval | query: {q}",
    "Classification": "task: classification | query: {q}",
    "Clustering": "task: clustering | query: {q}",
    "SentenceSimilarity": "task: sentence similarity | query: {q}",
    "Document": None,  # use prompt_name="Document" -> applies "title: none"
}


def load_model(model_id: str = MODEL_ID) -> SentenceTransformer:
    """Load text+vision only (drop 300M audio encoder). CPU float32, no CUDA/float16."""
    # Model card: NEVER use float16 (NaN / silent degradation). bf16 only on CUDA.
    # Intel iGPU / CPU host -> float32.
    model = SentenceTransformer(
        model_id,
        config_kwargs={"audio_config": None},  # text+image = ~440M, no audio
        model_kwargs={"torch_dtype": torch.float32},
        device="cpu",
    )
    model.eval()
    return model


def _check_emb(emb) -> None:
    import numpy as np
    arr = emb if isinstance(emb, __import__("numpy").ndarray) else np.asarray(emb)
    assert arr.ndim == 2 and arr.shape[1] == OUTPUT_DIM, f"expected (*,{OUTPUT_DIM}), got {arr.shape}"
    norms = (arr ** 2).sum(axis=1) ** 0.5
    assert ((norms - 1.0).__abs__() < 0.05).all(), f"embeddings not L2-normalized: norms={norms[:4]}"


def embed_texts(model: SentenceTransformer, texts: list[str], prompt_name: str = "SearchQuery"):
    """Embed text queries/documents at 512d, L2-normalized."""
    emb = model.encode(
        texts,
        prompt_name=prompt_name,
        truncate_dim=OUTPUT_DIM,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _check_emb(emb)
    return emb


def embed_images(model: SentenceTransformer, images: list):
    """Embed images (PIL.Image, path str, or list thereof). No prefix applied."""
    # sentence-transformers accepts PIL images directly for vision models.
    embs = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        emb = model.encode(
            [img],
            truncate_dim=OUTPUT_DIM,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embs.append(emb[0])
    import numpy as np
    out = np.stack(embs)
    _check_emb(out)
    return out


def embed_interleaved(model: SentenceTransformer, text: str, images: list | None = None):
    """Joint text+image embedding. Text must contain one <|image|> per image, in order."""
    pils = []
    for img in images or []:
        pils.append(Image.open(img).convert("RGB") if isinstance(img, str) else img)
    payload = {"text": text, "image": pils} if pils else {"text": text}
    emb = model.encode(
        [payload],
        truncate_dim=OUTPUT_DIM,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _check_emb(emb)
    return emb


if __name__ == "__main__":
    print(f"Loading {MODEL_ID} (text+vision, CPU float32)...")
    m = load_model()
    q = ["What causes the northern lights?"]
    d = ["title: none | text: The northern lights are caused by charged particles from the sun."]
    qe = embed_texts(m, q, prompt_name="SearchQuery")
    de = embed_texts(m, d, prompt_name="Document")
    print("query shape:", qe.shape, "doc shape:", de.shape)
    print("similarity:", float(m.similarity(qe, de)[0][0]))
    print("OK - 512d normalized embeddings.")
