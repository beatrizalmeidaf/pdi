"""Conjuntos de dados. Todas as imagens saem já na resolução de entrada do modelo
(uint8 RGB), de modo que a técnica e o espectro operam exatamente sobre o que
a rede recebe.

- cifar10 / cifar100: 32x32 (torchvision).
- imagenette: subconjunto de 10 classes do ImageNet (fast.ai), redimensionado
  (lado menor 256) e recortado ao centro (224). Rótulos mapeados para os
  índices do ImageNet-1k, então os modelos pré-treinados são usados sem
  nenhuma adaptação de cabeça.
- imagenet_val: pasta local no formato <raiz>/<wnid>/*.JPEG (opcional).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# wnid -> índice no ImageNet-1k (ordem alfabética dos wnids)
IMAGENETTE = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888605": 701,  # parachute
}
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


@dataclass
class ImageSet:
    name: str
    ids: list[str]
    labels: np.ndarray            # rótulo no espaço de saída do modelo
    local_labels: np.ndarray      # 0..K-1 (para sondas lineares)
    loader: Callable[[int], np.ndarray]
    image_size: tuple[int, int]
    domain: str                   # imagenet | cifar10 | cifar100
    class_subset: list[int] | None = None  # índices do ImageNet presentes (imagenette)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def batches(self, batch_size: int) -> Iterator[tuple[list[str], np.ndarray, np.ndarray, np.ndarray]]:
        for s in range(0, len(self), batch_size):
            idx = range(s, min(s + batch_size, len(self)))
            imgs = np.stack([self.loader(i) for i in idx])
            yield [self.ids[i] for i in idx], imgs, self.labels[list(idx)], self.local_labels[list(idx)]

    def load_all(self) -> np.ndarray:
        return np.stack([self.loader(i) for i in range(len(self))])


def _subset(n_total: int, n: int | None, seed: int) -> np.ndarray:
    if n is None or n >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def _cifar(name: str, split: str, n, seed, data_dir: Path) -> ImageSet:
    from torchvision import datasets as tvd

    cls = tvd.CIFAR10 if name == "cifar10" else tvd.CIFAR100
    ds = cls(root=str(data_dir / name), train=(split == "train"), download=True)
    data, targets = ds.data, np.asarray(ds.targets)
    sel = _subset(len(data), n, seed)
    return ImageSet(
        name=name,
        ids=[f"{name}_{split}_{i}" for i in sel],
        labels=targets[sel],
        local_labels=targets[sel],
        loader=lambda i, d=data, s=sel: d[s[i]],
        image_size=(32, 32),
        domain=name,
    )


def _imagenet_loader(paths: list[Path], resize: int = 256, crop: int = 224):
    from PIL import Image

    def load(i: int) -> np.ndarray:
        im = Image.open(paths[i]).convert("RGB")
        w, h = im.size
        s = resize / min(w, h)
        im = im.resize((max(crop, round(w * s)), max(crop, round(h * s))), Image.BILINEAR)
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        return np.asarray(im.crop((l, t, l + crop, t + crop)))

    return load


def _folder(name: str, root: Path, wnid_to_idx: dict, split_tag: str, n, seed) -> ImageSet:
    if not root.exists():
        raise FileNotFoundError(f"{root} não existe. Rode `make download` (ou ajuste paths.imagenet_val).")
    files, labels, local = [], [], []
    wnids = sorted(wnid_to_idx)
    for k, wnid in enumerate(wnids):
        for p in sorted((root / wnid).glob("*")):
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                files.append(p)
                labels.append(wnid_to_idx[wnid])
                local.append(k)
    sel = _subset(len(files), n, seed)
    files = [files[i] for i in sel]
    return ImageSet(
        name=name,
        ids=[f"{name}_{split_tag}_{p.stem}" for p in files],
        labels=np.asarray(labels)[sel],
        local_labels=np.asarray(local)[sel],
        loader=_imagenet_loader(files),
        image_size=(224, 224),
        domain="imagenet",
        class_subset=sorted(set(wnid_to_idx.values())) if len(wnid_to_idx) < 1000 else None,
    )


def _synthetic(split: str, n, seed) -> ImageSet:
    """Imagens sintéticas 32x32 com rótulos aleatórios — só para testar o pipeline offline."""
    from .verify import synthetic_images

    n = n or 64
    imgs = synthetic_images(n, 32, seed=seed + (0 if split == "val" else 1))
    labels = np.random.default_rng(seed).integers(0, 10, n)
    return ImageSet("synthetic", [f"synthetic_{split}_{i}" for i in range(n)], labels, labels,
                    lambda i, d=imgs: d[i], (32, 32), "cifar10")


def load_dataset(name: str, split: str, n: int | None, seed: int, paths: dict) -> ImageSet:
    data_dir = Path(paths.get("data", "data"))
    if name == "synthetic":
        return _synthetic(split, n, seed)
    if name in ("cifar10", "cifar100"):
        return _cifar(name, split, n, seed, data_dir)
    if name == "imagenette":
        root = data_dir / "imagenette2-320" / ("train" if split == "train" else "val")
        return _folder(name, root, IMAGENETTE, split, n, seed)
    if name == "imagenet_val":
        root = Path(paths.get("imagenet_val", data_dir / "imagenet" / "val"))
        wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
        if split == "train":
            root = Path(paths.get("imagenet_train", data_dir / "imagenet" / "train"))
        return _folder(name, root, {w: i for i, w in enumerate(wnids)}, split, n, seed)
    raise ValueError(f"dataset desconhecido: {name}")
