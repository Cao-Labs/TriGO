from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import Tensor


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYUXO")


def sanitize_sequence(sequence: str) -> str:
    return "".join(aa if aa in VALID_AA else "X" for aa in sequence.upper())


def resolve_esm_model_path(model_path: str) -> str:
    """
    Resolve local ESM checkpoint path with practical fallbacks.

    Lookup order:
    1) explicit `model_path`
    2) env `ESM_MODEL_PATH`
    3) `~/.cache/torch/hub/checkpoints/<basename(model_path)>`
    """
    explicit_path = Path(model_path).expanduser()
    env_path = os.getenv("ESM_MODEL_PATH")
    cache_guess = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / explicit_path.name

    candidates = [explicit_path]
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(cache_guess)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"ESM checkpoint not found. Tried: {[str(p) for p in candidates]}. "
        "Set --esm-model-path or environment variable ESM_MODEL_PATH."
    )


def is_hf_esmc_model_path(model_path: str) -> bool:
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    architectures = config.get("architectures") or []
    return config.get("model_type") == "esmc" or any("ESMC" in name for name in architectures)


@dataclass
class ESMEncoderConfig:
    model_path: str = "esm/esm1b_t33_650M_UR50S.pt"
    repr_layer: int = 33
    device: str = "cuda"
    chunk_size: int = 1022
    use_fp16: bool = False


class ESMSequenceEncoder:
    """
    Local ESM encoder wrapper.
    """

    def __init__(self, config: ESMEncoderConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        resolved_model_path = resolve_esm_model_path(config.model_path)
        self.config.model_path = resolved_model_path

        if is_hf_esmc_model_path(resolved_model_path):
            self._init_hf_esmc(resolved_model_path)
        else:
            self._init_fair_esm(resolved_model_path)

    def _init_hf_esmc(self, resolved_model_path: str) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Package `transformers` with ESMC support is required for Hugging Face ESMC models."
            ) from exc

        self.backend = "hf_esmc"
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
        self.model = AutoModel.from_pretrained(resolved_model_path).eval().to(self.device)

        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        n_layers = int(getattr(self.model.config, "n_layers", 0) or getattr(self.model.config, "num_hidden_layers", 0))
        if n_layers > 0 and self.config.repr_layer == 33:
            self.repr_layer = n_layers
        else:
            self.repr_layer = self.config.repr_layer
        self.n_layers = n_layers
        self.embedding_dim = int(getattr(self.model.config, "d_model", 0) or getattr(self.model.config, "hidden_size", 0))
        self.chunk_size = self.config.chunk_size

    def _init_fair_esm(self, resolved_model_path: str) -> None:
        self.backend = "fair_esm"

        # PyTorch>=2.6 defaults torch.load(weights_only=True). Some fair-esm checkpoints
        # include argparse.Namespace in the serialized payload, so allowlist it.
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([argparse.Namespace])

        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "Package `esm` is required. Install fair-esm before running feature extraction."
            ) from exc
        if not hasattr(esm, "pretrained"):
            raise ImportError(
                "The installed `esm` package does not expose fair-esm pretrained loaders. "
                "Use a Hugging Face ESMC model directory, or install fair-esm in a separate environment "
                "for legacy .pt checkpoints."
            )

        model, alphabet = esm.pretrained.load_model_and_alphabet_local(resolved_model_path)
        self.model = model.eval().to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = self.config.repr_layer

        max_positions = getattr(self.model, "max_positions", None)
        if isinstance(max_positions, int) and max_positions > 2:
            self.chunk_size = min(self.config.chunk_size, max_positions - 2)
        else:
            self.chunk_size = self.config.chunk_size

        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def _encode_fair_esm_chunk(self, sequence: str) -> Tensor:
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        out = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer][0, 1 : len(sequence) + 1]
        return reps.float().cpu()

    @torch.no_grad()
    def _encode_hf_esmc_chunk(self, sequence: str) -> Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        use_hidden_states = 0 <= self.repr_layer < self.n_layers
        out = self.model(**inputs, output_hidden_states=use_hidden_states)
        if use_hidden_states:
            reps = out.hidden_states[self.repr_layer][0, 1 : len(sequence) + 1]
        else:
            reps = out.last_hidden_state[0, 1 : len(sequence) + 1]
        return reps.float().cpu()

    @torch.no_grad()
    def _encode_chunk(self, sequence: str) -> Tensor:
        if self.backend == "hf_esmc":
            return self._encode_hf_esmc_chunk(sequence)
        return self._encode_fair_esm_chunk(sequence)

    @torch.no_grad()
    def encode(self, sequence: str) -> Tensor:
        sequence = sanitize_sequence(sequence)
        if len(sequence) == 0:
            embedding_dim = getattr(self, "embedding_dim", 0)
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        if len(sequence) <= self.chunk_size:
            return self._encode_chunk(sequence)

        chunks: List[Tensor] = []
        for start in range(0, len(sequence), self.chunk_size):
            end = min(start + self.chunk_size, len(sequence))
            chunk = sequence[start:end]
            chunks.append(self._encode_chunk(chunk))
        return torch.cat(chunks, dim=0)
