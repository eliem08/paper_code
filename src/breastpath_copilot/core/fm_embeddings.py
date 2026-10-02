"""Pathology foundation-model embedding backends.

This module exposes a thin abstraction over public pathology foundation models
(UNI, Virchow2, Phikon-v2) so that downstream code (MoE gating, retrieval,
agentic tools) can stay backend-agnostic. Each backend lazily loads its model
weights so that the rest of the package can be imported on CPU-only / no-token
environments (e.g. CI, schema tests).

The interface intentionally mirrors :class:`breastpath_copilot.core.embeddings.EmbeddingBackend`
via :meth:`FMBackend.encode_patch`, so existing call sites that take an
``EmbeddingBackend`` can be passed an ``FMBackend`` without changes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, ClassVar, Iterable

import numpy as np
from PIL import Image

# Pathology RoIs / WSIs can be very large (some BRACS RoIs exceed 100 megapixels).
# These are trusted local files, so disable PIL's decompression-bomb guard to
# avoid a hard DecompressionBombError crashing a long extraction run. Without
# this, a single oversized tile (> ~178 MP) would abort the whole job.
Image.MAX_IMAGE_PIXELS = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for foundation-model backends. "
            "Install torch / torchvision / timm / transformers."
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_imagenet_transform(input_size: int) -> Callable[[Image.Image], "torch.Tensor"]:
    _require_torch()
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class FMBackend(ABC):
    """Abstract pathology foundation-model backend.

    Subclasses set :attr:`name`, :attr:`model_id`, :attr:`embedding_dim`,
    :attr:`input_size`, and implement :meth:`_load_model` and
    :meth:`_forward`. Model weights are loaded lazily on first use.
    """

    name: ClassVar[str]
    model_id: ClassVar[str]
    embedding_dim: ClassVar[int]
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = False
    license_note: ClassVar[str] = ""

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        fp16: bool = True,
        hf_token: str | None = None,
        local_checkpoint: str | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.fp16 = bool(fp16) and self.device.startswith("cuda")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        # Path to a fine-tuned checkpoint (a Phikon2Finetuner/FMFineTuneClassifier
        # state dict with 'backbone.*' keys). When set, those encoder weights are
        # loaded over the pretrained model so this backend re-extracts embeddings
        # from the fine-tuned encoder. Supported for ALL architectures.
        self.local_checkpoint = local_checkpoint
        self._model: "nn.Module | None" = None
        self._transform: Callable[[Image.Image], "torch.Tensor"] | None = None

    @abstractmethod
    def _load_model(self) -> "nn.Module":  # pragma: no cover - subclassed
        """Load and return the underlying torch model in eval mode."""

    def _build_transform(self) -> Callable[[Image.Image], "torch.Tensor"]:
        return _build_imagenet_transform(self.input_size)

    @abstractmethod
    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":  # pragma: no cover - subclassed
        """Run the model and return a (B, embedding_dim) feature tensor."""

    @staticmethod
    def _load_finetuned_encoder(model: "nn.Module", checkpoint_path: str) -> None:
        """Load encoder weights from a fine-tuner checkpoint into ``model``.

        The checkpoint's ``model_state_dict`` holds ``backbone.*`` (encoder) and
        ``classifier.*`` (head) keys; we strip the ``backbone.`` prefix and load
        only the encoder, ignoring the head.
        """
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        encoder_state = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        if not encoder_state:
            raise ValueError(
                f"No 'backbone.*' keys in checkpoint {checkpoint_path!r}; "
                "is this a fine-tuner checkpoint?"
            )
        model.load_state_dict(encoder_state, strict=False)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            if self.requires_hf_token and not self.hf_token:
                raise RuntimeError(
                    f"{self.name} requires HF_TOKEN with access to '{self.model_id}'. "
                    "Request access on the Hugging Face model page and set HF_TOKEN."
                )
            model = self._load_model()
            if self.local_checkpoint is not None:
                self._load_finetuned_encoder(model, self.local_checkpoint)
            model.eval()
            model.to(self.device)
            if self.fp16:
                model.half()
            self._model = model
            self._transform = self._build_transform()

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore[misc]
    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Encode an iterable of PIL images. Returns (N, D) float32 array."""
        _require_torch()
        self._ensure_loaded()
        assert self._transform is not None and self._model is not None

        feats: list[np.ndarray] = []
        buffer: list["torch.Tensor"] = []
        for img in images:
            buffer.append(self._transform(img))
            if len(buffer) >= self.batch_size:
                feats.append(self._encode_buffer(buffer))
                buffer.clear()
        if buffer:
            feats.append(self._encode_buffer(buffer))
        if not feats:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def _encode_buffer(self, buffer: list["torch.Tensor"]) -> np.ndarray:
        batch = torch.stack(buffer, dim=0).to(self.device)
        if self.fp16:
            batch = batch.half()
        out = self._forward(batch)
        return out.detach().float().cpu().numpy()

    def encode_paths(self, paths: Iterable[str | Path]) -> np.ndarray:
        return self.encode_images(_load_pil_rgb(p) for p in paths)

    def encode_patch(self, patch_path: str | Path) -> np.ndarray:
        """Single-patch convenience method matching EmbeddingBackend's signature."""
        return self.encode_paths([patch_path])[0]

    def encode_roi_multipatch(
        self,
        path: str | Path,
        *,
        patch_size: int = 224,
        max_patches: int = 16,
        tissue_only: bool = True,
        return_bag: bool = False,
    ) -> np.ndarray:
        """Tile one RoI into native-resolution patches, embed each, mean-pool.

        Returns a (D,) mean-pooled vector by default, or the full (K, D) bag
        when ``return_bag`` is True (for downstream attention-MIL).
        """
        from breastpath_copilot.core.roi_tiling import tile_roi_path

        patches = tile_roi_path(
            path, patch_size=patch_size, max_patches=max_patches, tissue_only=tissue_only
        )
        bag = self.encode_images(patches)  # (K, D)
        if return_bag:
            return bag
        if bag.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        return bag.mean(axis=0).astype(np.float32, copy=False)

    def encode_paths_multipatch(
        self,
        paths: Iterable[str | Path],
        *,
        patch_size: int = 224,
        max_patches: int = 16,
        tissue_only: bool = True,
    ) -> np.ndarray:
        """Mean-pooled multi-patch embedding for each RoI path. Returns (N, D)."""
        vecs = [
            self.encode_roi_multipatch(
                p, patch_size=patch_size, max_patches=max_patches, tissue_only=tissue_only
            )
            for p in paths
        ]
        if not vecs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(vecs).astype(np.float32, copy=False)


class UNIBackend(FMBackend):
    """UNI: general-purpose pathology foundation model (Chen et al., Nat. Med. 2024).

    HF model card: ``MahmoodLab/UNI`` (gated; request access).
    Architecture: ViT-L/16, 1024-d CLS token.
    """

    name: ClassVar[str] = "uni"
    model_id: ClassVar[str] = "MahmoodLab/UNI"
    embedding_dim: ClassVar[int] = 1024
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = True
    license_note: ClassVar[str] = "CC-BY-NC, gated access on HF"

    def _load_model(self) -> "nn.Module":
        import timm

        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(batch)


class Virchow2Backend(FMBackend):
    """Virchow2: clinical-grade pathology foundation model (Paige).

    HF model card: ``paige-ai/Virchow2`` (gated). Architecture: ViT-H/14.
    Pooling concatenates the CLS token with the mean of the patch tokens,
    matching the model card recommendation (output dim = 2 * hidden_size).
    """

    name: ClassVar[str] = "virchow2"
    model_id: ClassVar[str] = "paige-ai/Virchow2"
    embedding_dim: ClassVar[int] = 2560
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = True
    license_note: ClassVar[str] = "Apache-2.0 with use restrictions, gated on HF"

    def _load_model(self) -> "nn.Module":
        import timm
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        output = self._model(batch)
        cls_token = output[:, 0]
        patch_tokens = output[:, 5:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


class PhikonV2Backend(FMBackend):
    """Phikon-v2: public pathology FM from Owkin.

    HF model card: ``owkin/phikon-v2`` (public). Architecture: ViT-L/16, 1024-d.
    Used in the hybrid strategy as the *fine-tunable* expert (frozen by default,
    set :attr:`trainable=True` and pass to an optimizer to fine-tune downstream).
    """

    name: ClassVar[str] = "phikon_v2"
    model_id: ClassVar[str] = "owkin/phikon-v2"
    embedding_dim: ClassVar[int] = 1024
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = False
    license_note: ClassVar[str] = "Owkin non-commercial research license"

    def __init__(self, *args, trainable: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = bool(trainable)

    def _load_model(self) -> "nn.Module":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_id, token=self.hf_token)
        if not self.trainable:
            for p in model.parameters():
                p.requires_grad_(False)
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        outputs = self._model(pixel_values=batch)
        return outputs.last_hidden_state[:, 0]


class ResNet50Backend(FMBackend):
    """ResNet50 baseline (ImageNet pretrained)."""
    name: ClassVar[str] = "resnet50"
    model_id: ClassVar[str] = "torchvision/resnet50"
    embedding_dim: ClassVar[int] = 2048
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = False
    license_note: ClassVar[str] = "MIT (ImageNet pretrained)"

    def _load_model(self) -> "nn.Module":
        from torchvision import models
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(batch)


class DenseNet121Backend(FMBackend):
    """DenseNet121 baseline (ImageNet pretrained)."""
    name: ClassVar[str] = "densenet121"
    model_id: ClassVar[str] = "torchvision/densenet121"
    embedding_dim: ClassVar[int] = 1024
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = False
    license_note: ClassVar[str] = "MIT (ImageNet pretrained)"

    def _load_model(self) -> "nn.Module":
        from torchvision import models
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(batch)


class ViTBaseBackend(FMBackend):
    """ViT-Base/16 baseline (ImageNet pretrained)."""
    name: ClassVar[str] = "vit_base"
    model_id: ClassVar[str] = "timm/vit_base_patch16_224"
    embedding_dim: ClassVar[int] = 768
    input_size: ClassVar[int] = 224
    requires_hf_token: ClassVar[bool] = False
    license_note: ClassVar[str] = "Apache-2.0 (ImageNet pretrained)"

    def _load_model(self) -> "nn.Module":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        return model

    def _forward(self, batch: "torch.Tensor") -> "torch.Tensor":
        assert self._model is not None
        return self._model(batch)


BACKEND_REGISTRY: dict[str, type[FMBackend]] = {
    UNIBackend.name: UNIBackend,
    Virchow2Backend.name: Virchow2Backend,
    PhikonV2Backend.name: PhikonV2Backend,
    ResNet50Backend.name: ResNet50Backend,
    DenseNet121Backend.name: DenseNet121Backend,
    ViTBaseBackend.name: ViTBaseBackend,
}


def build_backend(name: str, **kwargs) -> FMBackend:
    """Factory: build an FM backend by short name (``uni``, ``virchow2``, ``phikon_v2``)."""
    key = name.lower()
    if key not in BACKEND_REGISTRY:
        raise KeyError(
            f"Unknown FM backend {name!r}. Available: {sorted(BACKEND_REGISTRY)}"
        )
    return BACKEND_REGISTRY[key](**kwargs)
