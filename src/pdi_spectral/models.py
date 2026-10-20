"""Classificadores pré-treinados, sempre em ``model.eval()`` e ``torch.no_grad()``.

ImageNet: pesos oficiais do torchvision (V1, para bater com os números clássicos).
CIFAR: pesos públicos de ``chenyaofo/pytorch-cifar-models`` (torch.hub).
"""
from __future__ import annotations

import copy

import numpy as np

ZOO: dict[str, dict] = {
    # ImageNet-1k (torchvision)
    "resnet18": dict(src="tv", fn="resnet18", weights="IMAGENET1K_V1", family="cnn", domain="imagenet"),
    "resnet50": dict(src="tv", fn="resnet50", weights="IMAGENET1K_V1", family="cnn", domain="imagenet"),
    "vgg16_bn": dict(src="tv", fn="vgg16_bn", weights="IMAGENET1K_V1", family="cnn", domain="imagenet"),
    "efficientnet_b0": dict(src="tv", fn="efficientnet_b0", weights="IMAGENET1K_V1", family="cnn", domain="imagenet"),
    "convnext_tiny": dict(src="tv", fn="convnext_tiny", weights="IMAGENET1K_V1", family="cnn", domain="imagenet"),
    "vit_b_16": dict(src="tv", fn="vit_b_16", weights="IMAGENET1K_V1", family="vit", domain="imagenet"),
    "swin_t": dict(src="tv", fn="swin_t", weights="IMAGENET1K_V1", family="vit", domain="imagenet"),
}
for _ds in ("cifar10", "cifar100"):
    for _arch in ("resnet20", "resnet56", "vgg16_bn", "mobilenetv2_x1_0", "repvgg_a0"):
        ZOO[f"{_ds}_{_arch}"] = dict(src="hub", repo="chenyaofo/pytorch-cifar-models", fn=f"{_ds}_{_arch}",
                                     family="cnn", domain=_ds)

NORM = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def build_module(name: str):
    import torch
    import torchvision

    spec = ZOO[name]
    if spec["src"] == "tv":
        return getattr(torchvision.models, spec["fn"])(weights=spec["weights"])
    return torch.hub.load(spec["repo"], spec["fn"], pretrained=True, trust_repo=True, verbose=False)


class Classifier:
    def __init__(self, name: str, device: str = "cpu", module=None):
        import torch

        self.name = name
        self.spec = ZOO[name]
        self.family = self.spec["family"]
        self.domain = self.spec["domain"]
        self.device = device
        self.net = (module if module is not None else build_module(name)).to(device).eval()
        mean, std = NORM[self.domain]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self._feat = None
        last = [m for m in self.net.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.register_forward_hook(lambda m, inp, out: setattr(self, "_feat", inp[0].detach()))

    # ------------------------------------------------------------------
    def _tensor(self, imgs: np.ndarray):
        import torch

        x = torch.from_numpy(np.ascontiguousarray(imgs)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def logits(self, imgs: np.ndarray, batch_size: int = 128) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for s in range(0, len(imgs), batch_size):
                out.append(self.net(self._tensor(imgs[s : s + batch_size])).float().cpu().numpy())
        return np.concatenate(out)

    def features(self, imgs: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Entrada da última camada linear (representação penúltima)."""
        import torch

        out = []
        with torch.no_grad():
            for s in range(0, len(imgs), batch_size):
                self.net(self._tensor(imgs[s : s + batch_size]))
                out.append(self._feat.flatten(1).float().cpu().numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------------
    @property
    def has_bn(self) -> bool:
        import torch

        return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.net.modules())

    def bn_adapted(self, imgs: np.ndarray, batch_size: int = 128) -> "Classifier":
        """Cópia com as estatísticas de BatchNorm re-estimadas em ``imgs`` (sem gradiente).

        Separa a parcela de *domain shift* de 1ª/2ª ordem das ativações sem
        tocar em nenhum peso aprendido.
        """
        import torch

        net = copy.deepcopy(self.net).eval()
        for m in net.modules():  # o deepcopy leva junto o hook do modelo original
            m._forward_hooks.clear()
        bns =[m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None  # média cumulativa
            m.train()
        clf = Classifier(self.name, self.device, module=net)
        with torch.no_grad():
            for s in range(0, len(imgs), batch_size):
                net(clf._tensor(imgs[s : s + batch_size]))
        net.eval()
        return clf


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
