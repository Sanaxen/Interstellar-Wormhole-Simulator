from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderConfig:
    entrance_texture: Path
    exit_texture: Path
    output_dir: Path
    width: int = 960
    height: int = 540
    frames: int = 240
    fps: int = 30
    use_gpu: bool = False
    gpu_backend: str = "cpu"
    rho: float = 1.0
    a: float = 3.0
    camera_distance: float = 12.0
    mass_parameter: float = 1.0
    lensing_width: float = 1.0
    celestial_distance: float = 60.0
    geodesic_steps: int = 900
    antialias_samples: int = 4
    high_order_filter: bool = False
    cinematic_tunnel: bool = False
    ring_sharpness: float = 0.18
    roll_degrees: float = 0.0
    fov_degrees: float = 78.0
    turn_fraction: float = 0.28
    video_name: str = "wormhole_flythrough.mp4"
