from __future__ import annotations

import math

import numpy as np


def shape_radius(
    l_coord: np.ndarray | float,
    rho: float,
    a: float,
    mass: float,
    lensing_width: float | None = None,
) -> np.ndarray | float:
    """Interstellar paper's three-parameter wormhole shape function r(l)."""
    mass = max(float(mass), 1e-6)
    width = max(float(lensing_width if lensing_width is not None else mass), 1e-6)
    abs_l = np.abs(l_coord)
    x = 2.0 * (abs_l - a) / (math.pi * width)
    exterior = rho + width * (x * np.arctan(x) - 0.5 * np.log1p(x * x))
    return np.where(abs_l < a, rho, exterior)


def trace_to_celestial_spheres(
    rays: np.ndarray,
    camera_l: float,
    rho: float,
    a: float,
    mass: float,
    lensing_width: float,
    sphere_l: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace camera rays through the ultrastatic wormhole metric.

    Metric:
        ds^2 = -dt^2 + dl^2 + r(l)^2(dtheta^2 + sin(theta)^2 dphi^2)

    For null rays in an ultrastatic metric, the spatial path is a geodesic of
    dl^2 + r(l)^2 dOmega^2. Spherical symmetry lets each ray be integrated in
    the two-plane spanned by the local l-axis and the ray's transverse
    direction. The conserved impact parameter is b = r(l_camera) sin(alpha).
    """
    height, width = rays.shape[:2]
    flat = rays.reshape((-1, 3)).astype(np.float32)

    r0 = float(shape_radius(camera_l, rho, a, mass, lensing_width))
    mu0 = np.clip(flat[:, 2], -1.0, 1.0)
    transverse = flat[:, :2].copy()
    transverse_norm = np.maximum(np.linalg.norm(transverse, axis=1), 1e-8)
    transverse_unit = transverse / transverse_norm[:, None]

    impact = np.clip(r0 * np.sqrt(np.maximum(0.0, 1.0 - mu0 * mu0)), 0.0, r0)
    direction = np.where(mu0 >= 0.0, 1.0, -1.0).astype(np.float32)
    start_l = np.full(flat.shape[0], camera_l, dtype=np.float32)
    phi = np.zeros(flat.shape[0], dtype=np.float32)
    end_l = np.zeros(flat.shape[0], dtype=np.float32)

    outside = abs(camera_l) >= a
    if outside:
        camera_side = 1.0 if camera_l >= 0.0 else -1.0
        moving_toward_throat = direction * camera_side < 0.0
        passes_throat = impact < rho * (1.0 - 1e-5)

        direct = ~moving_toward_throat
        if np.any(direct):
            end_l[direct] = camera_side * sphere_l
            phi[direct] = _integrate_phi_along_l(
                start_l[direct],
                end_l[direct],
                impact[direct],
                rho,
                a,
                mass,
                lensing_width,
                steps,
            )

        through = moving_toward_throat & passes_throat
        if np.any(through):
            end_l[through] = -camera_side * sphere_l
            phi[through] = _integrate_phi_along_l(
                start_l[through],
                end_l[through],
                impact[through],
                rho,
                a,
                mass,
                lensing_width,
                steps,
            )

        turning = moving_toward_throat & ~passes_throat
        if np.any(turning):
            turn_abs_l = _turning_abs_l(
                impact[turning],
                np.abs(camera_l),
                rho,
                a,
                mass,
                lensing_width,
            )
            turn_l = (camera_side * turn_abs_l).astype(np.float32)
            end_l[turning] = camera_side * sphere_l
            phi[turning] = _integrate_phi_along_l(
                start_l[turning],
                turn_l,
                impact[turning],
                rho,
                a,
                mass,
                lensing_width,
                steps,
            )
            phi[turning] += _integrate_phi_along_l(
                turn_l,
                end_l[turning],
                impact[turning],
                rho,
                a,
                mass,
                lensing_width,
                steps,
            )
    else:
        end_l = direction * sphere_l
        phi = _integrate_phi_along_l(start_l, end_l, impact, rho, a, mass, lensing_width, steps)

    side = end_l >= 0.0
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sample_dirs = np.zeros_like(flat)
    sample_dirs[:, :2] = transverse_unit * sin_phi[:, None]
    sample_dirs[:, 2] = np.where(side, 1.0, -1.0) * cos_phi
    sample_dirs /= np.maximum(np.linalg.norm(sample_dirs, axis=1, keepdims=True), 1e-8)

    return side.reshape((height, width)), sample_dirs.reshape((height, width, 3)), phi.reshape((height, width))


def _integrate_phi_along_l(
    start_l: np.ndarray,
    end_l: np.ndarray,
    impact: np.ndarray,
    rho: float,
    a: float,
    mass: float,
    lensing_width: float,
    steps: int,
) -> np.ndarray:
    count = max(int(steps), 8)
    total = np.zeros_like(impact, dtype=np.float32)
    span = end_l - start_l
    abs_step = np.abs(span) / count
    active = abs_step > 1e-8
    if not np.any(active):
        return total

    for i in range(count):
        t = (i + 0.5) / count
        l_mid = start_l + span * t
        r_mid = np.maximum(shape_radius(l_mid, rho, a, mass, lensing_width).astype(np.float32), rho)
        radial_sq = np.maximum(1.0 - (impact / np.maximum(r_mid, 1e-6)) ** 2, 1e-7)
        integrand = impact / np.maximum(r_mid * r_mid * np.sqrt(radial_sq), 1e-6)
        total += np.where(active, integrand * abs_step, 0.0).astype(np.float32)
    return total


def _turning_abs_l(
    impact: np.ndarray,
    camera_abs_l: float,
    rho: float,
    a: float,
    mass: float,
    lensing_width: float,
) -> np.ndarray:
    low = np.full_like(impact, a, dtype=np.float32)
    high = np.full_like(impact, max(camera_abs_l, a), dtype=np.float32)
    for _ in range(32):
        mid = (low + high) * 0.5
        r_mid = shape_radius(mid, rho, a, mass, lensing_width).astype(np.float32)
        high = np.where(r_mid >= impact, mid, high)
        low = np.where(r_mid < impact, mid, low)
    return high
