from __future__ import annotations

import math
import shutil
import threading
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter

from .config import RenderConfig
from .geodesic import trace_to_celestial_spheres
from .panorama import Panorama


ProgressCallback = Callable[[int, int, Path], None]


class RenderCancelled(Exception):
    pass


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def look_at_rotation(position: np.ndarray, target: np.ndarray, roll_degrees: float = 0.0) -> np.ndarray:
    forward = target - position
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(forward, right)

    roll = math.radians(roll_degrees)
    if abs(roll) > 1e-5:
        cr, sr = math.cos(roll), math.sin(roll)
        right, up = right * cr + up * sr, -right * sr + up * cr
    return np.stack([right, up, forward], axis=1)


def rotation_from_forward(forward: np.ndarray, roll_degrees: float = 0.0) -> np.ndarray:
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(forward, right)
    roll = math.radians(roll_degrees)
    if abs(roll) > 1e-5:
        cr, sr = math.cos(roll), math.sin(roll)
        right, up = right * cr + up * sr, -right * sr + up * cr
    return np.stack([right, up, forward], axis=1)


def slerp_unit(a_vec: np.ndarray, b_vec: np.ndarray, t: float) -> np.ndarray:
    a_vec = a_vec / max(float(np.linalg.norm(a_vec)), 1e-8)
    b_vec = b_vec / max(float(np.linalg.norm(b_vec)), 1e-8)
    dot = float(np.clip(np.dot(a_vec, b_vec), -1.0, 1.0))
    if dot < -0.999:
        axis = np.cross(a_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a_vec, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        angle = math.pi * t
        return a_vec * math.cos(angle) + np.cross(axis, a_vec) * math.sin(angle)
    if dot > 0.999:
        return a_vec * (1.0 - t) + b_vec * t
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - t) * theta) * a_vec + math.sin(t * theta) * b_vec) / sin_theta


def turn_rotation(q: float, roll_degrees: float = 0.0) -> np.ndarray:
    angle = math.pi * q
    forward = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    roll = math.radians(roll_degrees)
    if abs(roll) > 1e-5:
        cr, sr = math.cos(roll), math.sin(roll)
        right, up = right * cr + up * sr, -right * sr + up * cr
    return np.stack([right, up, forward], axis=1)


def make_camera_rays(width: int, height: int, fov_degrees: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    aspect = width / height
    fov = math.radians(fov_degrees)
    y = 1.0 - ((np.arange(height, dtype=np.float32) + 0.5 + offset_y) / height) * 2.0
    x = (((np.arange(width, dtype=np.float32) + 0.5 + offset_x) / width) * 2.0 - 1.0) * aspect
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, 1.0 / math.tan(fov * 0.5))
    return normalize(np.stack([xx, yy, zz], axis=-1))


def camera_rotation(frame: int, total_frames: int, cfg: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    t = frame / max(total_frames - 1, 1)
    approach_end = 0.24
    tunnel_end = 0.72
    exit_glide_end = 0.84
    exit_z = cfg.a + cfg.camera_distance * 0.42
    if t < approach_end:
        q = smoothstep(0.0, approach_end, t)
        z = -cfg.camera_distance * (1.0 - q) + (-cfg.a * 0.92) * q
        position = np.array([0.0, 0.0, z], dtype=np.float32)
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elif t < tunnel_end:
        q = smoothstep(approach_end, tunnel_end, t)
        z = (-cfg.a * 0.92) * (1.0 - q) + (cfg.a * 0.92) * q
        position = np.array([0.0, 0.0, z], dtype=np.float32)
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elif t < exit_glide_end:
        q = smoothstep(tunnel_end, exit_glide_end, t)
        z = (cfg.a * 0.92) * (1.0 - q) + exit_z * q
        position = np.array([0.0, 0.0, z], dtype=np.float32)
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        q = smoothstep(exit_glide_end, 1.0, t)
        orbit_radius = cfg.rho * 0.36
        position = np.array(
            [
                math.sin(q * math.pi) * orbit_radius,
                0.0,
                exit_z,
            ],
            dtype=np.float32,
        )
        return position, turn_rotation(q, cfg.roll_degrees)
    return position, rotation_from_forward(forward, cfg.roll_degrees)


class WormholeRenderer:
    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self.gpu_renderer = None
        if cfg.use_gpu and cfg.gpu_backend.lower() == "opengl":
            try:
                from .opengl_renderer import OpenGLFrameRenderer

                self.gpu_renderer = OpenGLFrameRenderer(cfg)
            except Exception as exc:
                print(f"OpenGL renderer unavailable, falling back to CPU: {exc}")
        self.entrance = Panorama(cfg.entrance_texture, (24, 36, 82))
        self.exit = Panorama(cfg.exit_texture, (78, 38, 70))
        self.ray_cache: dict[tuple[float, float], np.ndarray] = {}
        yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float32)
        self.screen_radius = np.sqrt(
            ((xx - (cfg.width - 1) * 0.5) / cfg.height) ** 2
            + ((yy - (cfg.height - 1) * 0.5) / cfg.height) ** 2
        )

    def render_frame(self, frame: int) -> Image.Image:
        if self.gpu_renderer is not None:
            return self.gpu_renderer.render_frame(frame)
        cfg = self.cfg
        position, rotation = camera_rotation(frame, cfg.frames, cfg)
        offsets = self._sample_offsets(cfg.antialias_samples)
        accum = np.zeros((cfg.height, cfg.width, 3), dtype=np.float32)
        for offset_x, offset_y in offsets:
            local_rays = self._local_rays(offset_x, offset_y)
            rays = local_rays @ rotation.T
            rays = normalize(rays)
            accum += self._trace_sample(rays, float(position[2]))
        frame_img = accum / len(offsets)
        frame_img = self._add_vignette(frame_img)
        return Image.fromarray(np.clip(frame_img * 255.0, 0, 255).astype(np.uint8), "RGB")

    def _trace_sample(self, rays: np.ndarray, camera_l: float) -> np.ndarray:
        cfg = self.cfg
        rho = max(cfg.rho, 0.05)
        a = max(cfg.a, 0.001)
        mass = max(cfg.mass_parameter, 0.001)
        lensing_width = max(cfg.lensing_width, 0.001)
        sphere_l = max(cfg.celestial_distance, cfg.camera_distance + 2.0 * a + 1.0)
        side, sample_dirs, phi = trace_to_celestial_spheres(
            rays=rays,
            camera_l=camera_l,
            rho=rho,
            a=a,
            mass=mass,
            lensing_width=lensing_width,
            sphere_l=sphere_l,
            steps=cfg.geodesic_steps,
        )

        entrance_color = self.entrance.sample(sample_dirs)
        exit_color = self.exit.sample(sample_dirs)
        color = np.where(side[..., None], exit_color, entrance_color)
        if cfg.high_order_filter:
            color = self._filter_high_order(color, phi)
        if cfg.cinematic_tunnel:
            transition_width = max(a * 0.12, rho * 0.55)
            tunnel_weight = 1.0 - smoothstep(a, a + transition_width, abs(camera_l))
            if abs(camera_l) < a:
                tunnel_weight = 1.0
            if tunnel_weight > 0.0:
                tunnel = self._render_cinematic_tunnel(rays, camera_l, rho, a)
                color = color * (1.0 - tunnel_weight) + tunnel * tunnel_weight
        return color

    def _render_cinematic_tunnel(self, rays: np.ndarray, camera_l: float, rho: float, a: float) -> np.ndarray:
        forward = np.maximum(rays[..., 2], 0.05)
        depth_seed = 1.0 / np.maximum(np.sqrt((rays[..., 0] / forward) ** 2 + (rays[..., 1] / forward) ** 2), 0.035)
        bend_x = 0.16 * np.sin(depth_seed * 0.42 + camera_l * 0.32)
        bend_y = 0.10 * np.cos(depth_seed * 0.36 + camera_l * 0.27)
        x_over_z = rays[..., 0] / forward - bend_x
        y_over_z = rays[..., 1] / forward - bend_y
        radial = np.sqrt(x_over_z * x_over_z + y_over_z * y_over_z)
        theta = np.arctan2(y_over_z, x_over_z)
        depth = (1.0 / np.maximum(radial, 0.035)) + (camera_l + a) * 0.35
        tunnel_phase = np.clip((camera_l + a) / max(2.0 * a, 1e-6), 0.0, 1.0)
        wall_depth = depth + tunnel_phase * 5.2

        u = (theta / (2.0 * math.pi) + 0.5 + wall_depth * 0.028 + 0.018 * np.sin(wall_depth * 0.55)) % 1.0
        v = (wall_depth * 0.145 + 0.055 * np.sin(theta + wall_depth * 0.45)) % 1.0
        u2 = (theta / (2.0 * math.pi) + 0.5 - wall_depth * 0.018 + 0.027 * np.cos(theta + wall_depth * 0.31)) % 1.0
        v2 = (wall_depth * 0.105 + 0.045 * np.cos(theta - wall_depth * 0.37)) % 1.0
        theta_a = theta + (2.0 * math.pi / 3.0)
        theta_b = theta - (2.0 * math.pi / 3.0)
        u3 = (theta_a / (2.0 * math.pi) + 0.5 + wall_depth * 0.020) % 1.0
        v3 = (wall_depth * 0.130 + 0.040 * np.sin(theta_a + wall_depth * 0.41)) % 1.0
        u4 = (theta_b / (2.0 * math.pi) + 0.5 - wall_depth * 0.014) % 1.0
        v4 = (wall_depth * 0.118 + 0.036 * np.cos(theta_b - wall_depth * 0.33)) % 1.0
        wall = self._sample_pano_uv(self.exit, u, v)
        wall_alt = self._sample_pano_uv(self.exit, u2, v2)
        wall_ring = self._sample_pano_uv(self.exit, u3, v3) * 0.5 + self._sample_pano_uv(self.exit, u4, v4) * 0.5
        wall = wall * 0.56 + wall_alt * 0.24 + wall_ring * 0.20

        forward_dirs = rays.copy()
        forward_dirs[..., 2] = np.maximum(forward_dirs[..., 2], 0.08)
        forward_dirs = normalize(forward_dirs)
        exit_view = self.exit.sample(forward_dirs)
        entrance_echo = self.entrance.sample(-forward_dirs)

        early_shrink = smoothstep(0.01, 0.05, tunnel_phase)
        exit_approach = smoothstep(0.25, 0.95, tunnel_phase)
        far_core = 0.20 * (1.0 - exit_approach) + 0.36 * exit_approach
        core_inner = 0.52 * (1.0 - early_shrink) + far_core * early_shrink
        core_outer = core_inner + (0.12 * (1.0 - early_shrink) + (0.08 + 0.04 * exit_approach) * early_shrink)
        rim_radius = core_outer - 0.04
        rim_width = 0.075 * (1.0 - early_shrink) + (0.040 + 0.025 * exit_approach) * early_shrink

        aperture = 1.0 - smoothstep(core_inner * 0.38, core_inner * 0.95, radial)
        wall_mix = smoothstep(0.24, 0.42, radial)
        longitudinal_glow = 0.5 + 0.5 * np.cos(wall_depth * math.pi * 1.4)
        rib_phase = np.abs((wall_depth * 0.62) % 1.0 - 0.5) * 2.0
        ribs = 0.42 + 0.58 * smoothstep(0.20, 0.55, rib_phase)
        side_shade = 0.72 + 0.28 * np.clip(radial, 0.0, 1.0)
        tunnel_tint = np.array([0.14, 0.18, 0.24], dtype=np.float32)
        wall_avg = np.mean(wall, axis=-1, keepdims=True)
        wall = wall * 0.10 + wall_avg * 0.62 + tunnel_tint * 0.28
        wall = np.clip(wall * (0.72 + 0.28 * ribs[..., None]) * side_shade[..., None] * 1.04, 0.0, 1.0)
        rim = np.exp(-np.square((radial - rim_radius) / rim_width))[..., None]
        rim_color = exit_view * 0.55 + np.array([0.72, 0.86, 1.0], dtype=np.float32) * 0.45
        core = exit_view
        circular_core = 1.0 - smoothstep(core_inner, core_outer, radial)
        color = wall * (1.0 - circular_core[..., None]) + core * circular_core[..., None]
        color = color * (1.0 - rim * 0.45) + rim_color * rim * 0.45
        color = color * (1.0 - aperture[..., None] * 0.04) + exit_view * aperture[..., None] * 0.04
        return np.clip(color * 1.08, 0.0, 1.0)

    @staticmethod
    def _sample_pano_uv(panorama: Panorama, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        image = panorama.image
        height, width = image.shape[:2]
        uu = (u % 1.0) * width
        vv = np.clip(v, 0.0, 1.0) * (height - 1)
        x0 = np.floor(uu).astype(np.int32) % width
        y0 = np.clip(np.floor(vv).astype(np.int32), 0, height - 1)
        x1 = (x0 + 1) % width
        y1 = np.clip(y0 + 1, 0, height - 1)
        fu = (uu - np.floor(uu))[..., None]
        fv = (vv - np.floor(vv))[..., None]
        c00 = image[y0, x0]
        c10 = image[y0, x1]
        c01 = image[y1, x0]
        c11 = image[y1, x1]
        top = c00 * (1.0 - fu) + c10 * fu
        bottom = c01 * (1.0 - fu) + c11 * fu
        return top * (1.0 - fv) + bottom * fv

    @staticmethod
    def _filter_high_order(color: np.ndarray, phi: np.ndarray) -> np.ndarray:
        weight = smoothstep(math.pi * 1.4, math.pi * 4.5, np.abs(phi))[..., None]
        if float(np.max(weight)) <= 0.001:
            return color
        image = Image.fromarray(np.clip(color * 255.0, 0, 255).astype(np.uint8), "RGB")
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=1.6)), dtype=np.float32) / 255.0
        filtered = color * (1.0 - weight) + blurred * weight
        attenuation = 1.0 - 0.28 * weight
        return np.clip(filtered * attenuation, 0.0, 1.0)

    def _local_rays(self, offset_x: float, offset_y: float) -> np.ndarray:
        key = (offset_x, offset_y)
        if key not in self.ray_cache:
            self.ray_cache[key] = make_camera_rays(self.cfg.width, self.cfg.height, self.cfg.fov_degrees, offset_x, offset_y)
        return self.ray_cache[key]

    @staticmethod
    def _sample_offsets(samples: int) -> list[tuple[float, float]]:
        if samples <= 1:
            return [(0.0, 0.0)]
        if samples <= 4:
            return [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        return [
            (-0.33, -0.33),
            (0.0, -0.33),
            (0.33, -0.33),
            (-0.33, 0.0),
            (0.0, 0.0),
            (0.33, 0.0),
            (-0.33, 0.33),
            (0.0, 0.33),
            (0.33, 0.33),
        ]

    def _add_vignette(self, color: np.ndarray) -> np.ndarray:
        radial = self.screen_radius[..., None]
        vignette = 1.0 - np.clip((radial - 0.42) / 0.38, 0.0, 1.0) * 0.38
        return np.clip(color * vignette, 0.0, 1.0)


def render_sequence(
    cfg: RenderConfig,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = cfg.output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = WormholeRenderer(cfg)
    frame_paths: list[Path] = []
    for i in range(cfg.frames):
        if cancel_event and cancel_event.is_set():
            raise RenderCancelled("Render cancelled")
        image = renderer.render_frame(i)
        frame_path = frames_dir / f"frame_{i + 1:05d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)
        if progress:
            progress(i + 1, cfg.frames, frame_path)

    video_path = cfg.output_dir / cfg.video_name
    if cancel_event and cancel_event.is_set():
        raise RenderCancelled("Render cancelled")
    with imageio.get_writer(video_path, fps=cfg.fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for path in frame_paths:
            if cancel_event and cancel_event.is_set():
                raise RenderCancelled("Render cancelled")
            writer.append_data(imageio.imread(path))
    return video_path
