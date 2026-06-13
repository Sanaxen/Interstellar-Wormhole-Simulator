from __future__ import annotations

import argparse
from pathlib import Path

from .config import RenderConfig
from .renderer import render_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an Interstellar-inspired wormhole fly-through.")
    parser.add_argument("--entrance", required=True, type=Path, help="Entrance-side 360 panorama image")
    parser.add_argument("--exit", required=True, type=Path, help="Exit-side 360 panorama image")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=["cpu", "opengl"], default="cpu", help="Rendering backend")
    parser.add_argument("--rho", type=float, default=1.0, help="Paper shape parameter rho, the throat radius")
    parser.add_argument("--a", type=float, default=3.0, help="Paper shape parameter a, half-length of the cylindrical tunnel")
    parser.add_argument("--throat-diameter", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--tunnel-length", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--camera-distance", type=float, default=12.0)
    parser.add_argument("--mass-parameter", type=float, default=1.0, help="Paper shape parameter M")
    parser.add_argument("--lensing-width", "--w", dest="lensing_width", type=float, default=1.0, help="Lensing width parameter W")
    parser.add_argument("--celestial-distance", type=float, default=60.0, help="l coordinate of the texture spheres")
    parser.add_argument("--geodesic-steps", type=int, default=900, help="Numerical integration steps per frame")
    parser.add_argument("--antialias-samples", type=int, default=4, choices=[1, 4, 9], help="Subpixel geodesic samples per pixel")
    parser.add_argument("--high-order-filter", action=argparse.BooleanOptionalAction, default=False, help="Blur/dim high-order multiple images")
    parser.add_argument("--cinematic-tunnel", action=argparse.BooleanOptionalAction, default=False, help="Use abstract cylindrical tunnel visuals inside the wormhole")
    parser.add_argument("--ring-sharpness", type=float, default=0.18, help=argparse.SUPPRESS)
    parser.add_argument("--fov", type=float, default=78.0)
    parser.add_argument("--video-name", default="wormhole_flythrough.mp4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = RenderConfig(
        entrance_texture=args.entrance,
        exit_texture=args.exit,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        use_gpu=args.backend != "cpu",
        gpu_backend=args.backend,
        rho=args.rho if args.throat_diameter is None else args.throat_diameter * 0.5,
        a=args.a if args.tunnel_length is None else args.tunnel_length * 0.5,
        camera_distance=args.camera_distance,
        mass_parameter=args.mass_parameter,
        lensing_width=args.lensing_width,
        celestial_distance=args.celestial_distance,
        geodesic_steps=args.geodesic_steps,
        antialias_samples=args.antialias_samples,
        high_order_filter=args.high_order_filter,
        cinematic_tunnel=args.cinematic_tunnel,
        ring_sharpness=args.ring_sharpness,
        fov_degrees=args.fov,
        video_name=args.video_name,
    )

    def progress(done: int, total: int, _path: Path) -> None:
        print(f"\rRendering {done}/{total}", end="", flush=True)

    video_path = render_sequence(cfg, progress)
    print(f"\nDone: {video_path}")


if __name__ == "__main__":
    main()
