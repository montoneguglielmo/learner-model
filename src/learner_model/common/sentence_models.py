"""Sentence-embedding model helpers shared by the encoders (exercise_embed_v1, event_embed_v1)."""

import importlib.metadata as md


def resolve_revision(name, revision):
    """'main' changes over time: ask Hugging Face for the exact commit hash."""
    from huggingface_hub import HfApi
    return HfApi().model_info(name, revision=revision).sha


def load_model(m, revision):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        m["name"],
        revision=revision,
        device=m.get("device", "cuda"),
        model_kwargs={"torch_dtype": getattr(torch, m["dtype"])},
        processor_kwargs={"padding_side": "left"},  # recommended for Qwen3-Embedding (last-token pooling)
    )
    model.max_seq_length = m["max_length"]
    return model


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def library_versions():
    out = {}
    for lib in ["torch", "transformers", "sentence-transformers", "huggingface-hub", "polars", "numpy"]:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    return out
