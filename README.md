# Interstellar Wormhole Simulator

Windows-friendly Python app inspired by Oliver James, Eugenie von Tunzelmann,
Paul Franklin, and Kip S. Thorne, "Visualizing Interstellar's Wormhole"
(arXiv:1502.03809).

This project renders a configurable wormhole fly-through using two 360-degree
equirectangular panorama images:

- entrance-side universe texture
- exit-side universe texture

The renderer saves numbered frames, then combines them into an MP4 video.

## Quick Start on Windows

1. Double-click `setup.bat`.
2. Double-click `run_app.bat`.
3. Choose entrance and exit panorama images.
4. Set wormhole and animation parameters.
5. Click `Render Sequence`.
6. Watch generated frames in the preview panel. Use the frame slider to inspect
   any frame that has already been computed.

Click `Save Settings` to write the current GUI values to the fixed settings
file `wormhole_settings.json`. The GUI automatically loads this file on the
next launch, so the same panorama paths, output folder, backend, and wormhole
parameters are restored.

For command-line rendering, edit `render_cli.bat` or pass options directly:

```powershell
.\.venv\Scripts\python.exe -m wormhole_app.render_cli --help
```

For OpenGL GPU rendering, run `setup_opengl.bat`, then select `opengl` in the
GUI backend field or pass `--backend opengl` on the command line.

## Notes on the Model

The paper describes a general-relativistic visualization pipeline: combine the
camera frame with solutions of the null geodesic equation to map each camera
ray backward to one of two celestial spheres. This implementation follows that
workflow for the static spherically symmetric wormhole metric.

The implemented metric is:

```text
ds^2 = -dt^2 + dl^2 + r(l)^2(dtheta^2 + sin(theta)^2 dphi^2)
```

The implemented shape function is the paper's three-parameter form:

```text
r(l) = rho,                                             |l| < a
r(l) = rho + W[x atan(x) - 0.5 ln(1 + x^2)],            |l| >= a
x = 2(|l| - a) / (pi W)
```

For each camera ray, the renderer computes the conserved impact parameter
`b = r(l_camera) sin(alpha)` and integrates:

```text
dphi/dlambda = b / r(l)^2
dl/dlambda = +/- sqrt(1 - b^2 / r(l)^2)
```

The final angular coordinate selects the entrance or exit celestial sphere
texture.

The practical model includes:

- the ultrastatic wormhole metric
- the paper's three-parameter radius function `r(l)` using `rho`, `a`, and `M`
- per-pixel null-ray mapping by numerical geodesic integration
- direct UI controls for `rho`, `a`, `M`, and lensing width `W`
- numerical controls for celestial-sphere distance and integration steps
- subpixel antialiasing to reduce ring and caustic sampling artifacts
- optional high-order image filtering to blur and dim repeated multiple images
- optional cinematic tunnel interior mode that maps the exit panorama onto an
  abstract cylindrical passage while preserving geodesic rendering outside
- independent entrance and exit panoramas
- generated frame sequence and MP4 output
- live GUI preview of rendered frames with a computed-frame timeline slider
- optional OpenGL GPU backend for faster preview/rendering on non-NVIDIA GPUs

The default camera timing spends roughly half of the frame sequence inside the
wormhole throat, making cinematic tunnel passage frames slower and more
numerous than the approach and turn-back sections.

`W` controls the width of the transition from the cylindrical throat to the
asymptotically flat exterior. The coefficient and denominator both use `W`, so
the exterior approaches `dr/dl = 1` far from the wormhole. Small values localize
the lensing near the throat; large values spread the lensing over a wider area.

The intent is to make a useful Windows app that can render shots and explore
the paper's parameter effects. It is still a compact educational renderer, not
DNGR's production ray-bundle renderer.

## Files

- `setup.bat` - creates `.venv` and installs dependencies
- `run_app.bat` - launches the GUI
- `render_cli.bat` - example batch renderer
- `wormhole_app/gui.py` - Tkinter GUI
- `wormhole_app/renderer.py` - panorama sampling and wormhole renderer
- `wormhole_app/render_cli.py` - command-line entry point
