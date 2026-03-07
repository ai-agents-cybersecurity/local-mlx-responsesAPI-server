"""Model loading and holder singleton."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer, _download

log = logging.getLogger("mlx-server")


@dataclass
class ModelHolder:
    """Holds the loaded model + tokenizer in a thread-safe-ish way."""

    model_path: str = ""
    model: object = None
    tokenizer: object = None
    loaded: bool = False

    @staticmethod
    def _resolve_path(model_path: str) -> Path:
        """Return a local Path, using the HF cache if available to avoid network calls."""
        p = Path(model_path)
        if p.exists():
            return p
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / ("models--" + model_path.replace("/", "--"))
        if repo_dir.exists():
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                revisions = sorted(snapshots.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                for rev in revisions:
                    has_weights = any(rev.glob("*.safetensors"))
                    has_tokenizer = any(rev.glob("tokenizer*"))
                    if has_weights and has_tokenizer:
                        log.info("Using cached model at %s", rev)
                        return rev
            log.info("Incomplete cache for %s, downloading missing files...", model_path)
        return Path(_download(model_path))

    def load(self, model_path: str) -> None:
        log.info("Loading model %s ...", model_path)
        t0 = time.perf_counter()
        local_path = self._resolve_path(model_path)
        model, config = load_model(local_path, lazy=False, strict=False)
        tokenizer = load_tokenizer(
            local_path, eos_token_ids=config.get("eos_token_id", None)
        )
        self.model, self.tokenizer = model, tokenizer
        self.model_path = model_path
        self.loaded = True
        elapsed = time.perf_counter() - t0
        log.info("Model ready in %.1fs", elapsed)


# Global singleton
holder = ModelHolder()

# Serialize all inference -- MLX models are NOT thread-safe.
inference_lock = asyncio.Lock()
