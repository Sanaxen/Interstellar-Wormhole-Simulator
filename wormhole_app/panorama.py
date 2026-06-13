from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


class Panorama:
    def __init__(self, path: Path, fallback_tint: tuple[int, int, int] | None = None) -> None:
        self.path = Path(path)
        if self.path.exists():
            image = Image.open(self.path).convert("RGB")
        else:
            image = self._fallback_image(fallback_tint or (30, 40, 70))
        self.image = np.asarray(image, dtype=np.float32) / 255.0
        self.height, self.width = self.image.shape[:2]

    @staticmethod
    def _fallback_image(tint: tuple[int, int, int]) -> Image.Image:
        width, height = 2048, 1024
        y = np.linspace(0.0, 1.0, height)[:, None]
        x = np.linspace(0.0, 1.0, width)[None, :]
        base = np.zeros((height, width, 3), dtype=np.float32)
        base[..., 0] = tint[0] / 255.0 * (0.45 + 0.55 * (1.0 - y))
        base[..., 1] = tint[1] / 255.0 * (0.55 + 0.45 * x)
        base[..., 2] = tint[2] / 255.0 * (0.70 + 0.30 * np.sin(x * math.tau) ** 2)

        rng = np.random.default_rng(12345 + tint[0])
        for _ in range(1800):
            px = rng.integers(0, width)
            py = rng.integers(0, height)
            radius = rng.choice([1, 1, 1, 2])
            color = rng.uniform(0.65, 1.0)
            x0 = max(0, px - radius)
            x1 = min(width, px + radius + 1)
            y0 = max(0, py - radius)
            y1 = min(height, py + radius + 1)
            base[y0:y1, x0:x1] = np.maximum(base[y0:y1, x0:x1], color)
        return Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8), "RGB")

    def sample(self, dirs: np.ndarray) -> np.ndarray:
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]

        u = (np.arctan2(x, z) / (2.0 * math.pi) + 0.5) * self.width
        v = (0.5 - np.arcsin(np.clip(y, -1.0, 1.0)) / math.pi) * self.height

        x0 = np.floor(u).astype(np.int32) % self.width
        y0 = np.clip(np.floor(v).astype(np.int32), 0, self.height - 1)
        x1 = (x0 + 1) % self.width
        y1 = np.clip(y0 + 1, 0, self.height - 1)

        fu = (u - np.floor(u))[..., None]
        fv = (v - np.floor(v))[..., None]

        c00 = self.image[y0, x0]
        c10 = self.image[y0, x1]
        c01 = self.image[y1, x0]
        c11 = self.image[y1, x1]
        top = c00 * (1.0 - fu) + c10 * fu
        bottom = c01 * (1.0 - fu) + c11 * fu
        return top * (1.0 - fv) + bottom * fv

