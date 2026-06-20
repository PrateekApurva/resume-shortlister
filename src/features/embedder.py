"""
Sentence-transformer wrapper and work narrative builder.

all-MiniLM-L6-v2 produces 384-dimensional vectors. It runs on CPU,
is free (no API key), and handles English text well. At 100K candidates
batch encoding takes ~3-4 minutes on a modern CPU.

work_narrative uses ONLY career history descriptions — never the skills section.
This is intentional: embedding skills would let keyword-stuffers game the similarity
score without having done the actual work.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None   # lazy-loaded singleton


def get_model() -> SentenceTransformer:
    """Load the model once; reuse on subsequent calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Encode a list of strings → (N, 384) float32 numpy array.
    Vectors are L2-normalised so inner product = cosine similarity.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise so inner product == cosine sim
    )
    return embeddings.astype("float32")


def encode_single(text: str) -> np.ndarray:
    """Encode one string → (1, 384) float32 array, L2-normalised."""
    return encode_texts([text], show_progress=False)


def build_work_narrative(candidate: dict) -> str:
    """
    Build a text string from career history ONLY (no skills section).

    Format per job:
      Role: <title> at <company> (<type>, <industry>, <N> months)
      Work done: <description>

    company_type is 'consulting' or 'product' based on the consulting firms list.
    """
    from src.scoring.career import _is_consulting   # avoid circular import at module level

    parts = []
    for job in candidate["career_history"]:
        company_type = "consulting" if _is_consulting(job["company"]) else "product"
        description  = job.get("description", "").strip()
        if not description:
            description = "(no description provided)"
        parts.append(
            f"Role: {job['title']} at {job['company']} "
            f"({company_type}, {job.get('industry', 'unknown')}, {job.get('duration_months', 0)} months)\n"
            f"Work done: {description}"
        )

    return "\n\n---\n\n".join(parts) if parts else "(no career history)"
