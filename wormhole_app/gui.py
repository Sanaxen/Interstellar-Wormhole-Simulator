from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .config import RenderConfig
from .renderer import RenderCancelled, render_sequence


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Interstellar Wormhole Simulator")
        self.geometry("1180x760")
        self.minsize(1040, 680)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.frame_paths: list[Path] = []
        self.current_preview_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.vars = self._make_vars()
        self._build_ui()
        self.after(100, self._poll_queue)

    def _make_vars(self) -> dict[str, tk.Variable]:
        cwd = Path.cwd()
        return {
            "entrance": tk.StringVar(value=str(cwd / "entrance_panorama.bmp")),
            "exit": tk.StringVar(value=str(cwd / "exit_panorama.bmp")),
            "output": tk.StringVar(value=str(cwd / "render_output")),
            "width": tk.IntVar(value=960),
            "height": tk.IntVar(value=540),
            "frames": tk.IntVar(value=240),
            "fps": tk.IntVar(value=30),
            "backend": tk.StringVar(value="opengl"),
            "rho": tk.DoubleVar(value=2.0),
            "a": tk.DoubleVar(value=1.0),
            "camera_distance": tk.DoubleVar(value=12.0),
            "mass_parameter": tk.DoubleVar(value=1.0),
            "lensing_width": tk.DoubleVar(value=0.1),
            "celestial_distance": tk.DoubleVar(value=60.0),
            "geodesic_steps": tk.IntVar(value=900),
            "antialias_samples": tk.IntVar(value=4),
            "high_order_filter": tk.BooleanVar(value=False),
            "ring_sharpness": tk.DoubleVar(value=0.18),
            "fov": tk.DoubleVar(value=78.0),
            "frame_slider": tk.DoubleVar(value=1.0),
            "frame_label": tk.StringVar(value="Frame 0 / 0"),
            "status": tk.StringVar(value="Ready"),
        }

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=0, minsize=430)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.columnconfigure(0, weight=1)

        title = ttk.Label(left, text="Interstellar Wormhole Simulator", font=("Segoe UI", 17, "bold"))
        title.grid(row=0, column=0, sticky="w")

        files = ttk.LabelFrame(left, text="Panorama Textures", padding=12)
        files.grid(row=1, column=0, sticky="ew", pady=(14, 8))
        files.columnconfigure(1, weight=1)
        self._file_row(files, 0, "Entrance side", "entrance", save=False)
        self._file_row(files, 1, "Exit side", "exit", save=False)
        self._file_row(files, 2, "Output folder", "output", save=True)

        params = ttk.LabelFrame(left, text="Wormhole and Camera Parameters", padding=12)
        params.grid(row=2, column=0, sticky="ew", pady=8)
        for col in range(4):
            params.columnconfigure(col, weight=1)

        fields = [
            ("Width", "width", 0, 0),
            ("Height", "height", 0, 2),
            ("Frames", "frames", 1, 0),
            ("FPS", "fps", 1, 2),
            ("rho", "rho", 3, 0),
            ("a", "a", 3, 2),
            ("Camera distance", "camera_distance", 4, 0),
            ("FOV degrees", "fov", 4, 2),
            ("Mass parameter M", "mass_parameter", 5, 0),
            ("W lensing width", "lensing_width", 5, 2),
            ("Celestial distance", "celestial_distance", 6, 0),
            ("Geodesic steps", "geodesic_steps", 6, 2),
            ("AA samples", "antialias_samples", 7, 0),
        ]
        for label, key, row, col in fields:
            ttk.Label(params, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=5)
            ttk.Entry(params, textvariable=self.vars[key], width=14).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 16), pady=5
            )

        ttk.Label(params, text="Backend").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Combobox(
            params,
            textvariable=self.vars["backend"],
            values=("cpu", "opengl"),
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="ew", padx=(0, 16), pady=5)
        ttk.Checkbutton(
            params,
            text="High-order filter",
            variable=self.vars["high_order_filter"],
        ).grid(row=7, column=2, columnspan=2, sticky="w", pady=5)

        controls = ttk.Frame(left)
        controls.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure(2, weight=1)
        self.render_button = ttk.Button(controls, text="Render Sequence", command=self._start_render)
        self.render_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(controls, text="Cancel", command=self._cancel_render, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.progress = ttk.Progressbar(controls, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=12)
        ttk.Label(controls, textvariable=self.vars["status"]).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        preview = ttk.LabelFrame(root, text="Render Preview", padding=12)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview, anchor="center", text="No frames rendered yet")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", lambda _event: self._refresh_preview())

        timeline = ttk.Frame(preview)
        timeline.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        timeline.columnconfigure(0, weight=1)
        self.frame_slider = ttk.Scale(
            timeline,
            from_=1,
            to=1,
            variable=self.vars["frame_slider"],
            command=self._on_slider,
            state=tk.DISABLED,
        )
        self.frame_slider.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        ttk.Label(timeline, textvariable=self.vars["frame_label"], width=18).grid(row=0, column=1, sticky="e")

    def _file_row(self, parent: ttk.Frame, row: int, label: str, key: str, save: bool) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=5)
        command = lambda: self._browse(key, save)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", pady=5)

    def _browse(self, key: str, folder: bool) -> None:
        if folder:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ]
            )
        if path:
            self.vars[key].set(path)

    def _config_from_vars(self) -> RenderConfig:
        return RenderConfig(
            entrance_texture=Path(str(self.vars["entrance"].get())),
            exit_texture=Path(str(self.vars["exit"].get())),
            output_dir=Path(str(self.vars["output"].get())),
            width=max(160, int(self.vars["width"].get())),
            height=max(90, int(self.vars["height"].get())),
            frames=max(2, int(self.vars["frames"].get())),
            fps=max(1, int(self.vars["fps"].get())),
            use_gpu=str(self.vars["backend"].get()).lower() != "cpu",
            gpu_backend=str(self.vars["backend"].get()).lower(),
            rho=max(0.05, float(self.vars["rho"].get())),
            a=max(0.001, float(self.vars["a"].get())),
            camera_distance=max(0.1, float(self.vars["camera_distance"].get())),
            mass_parameter=max(0.001, float(self.vars["mass_parameter"].get())),
            lensing_width=max(0.001, float(self.vars["lensing_width"].get())),
            celestial_distance=max(5.0, float(self.vars["celestial_distance"].get())),
            geodesic_steps=max(20, int(self.vars["geodesic_steps"].get())),
            antialias_samples=1 if int(self.vars["antialias_samples"].get()) <= 1 else 4 if int(self.vars["antialias_samples"].get()) <= 4 else 9,
            high_order_filter=bool(self.vars["high_order_filter"].get()),
            ring_sharpness=max(0.02, float(self.vars["ring_sharpness"].get())),
            fov_degrees=min(140.0, max(20.0, float(self.vars["fov"].get()))),
        )

    def _start_render(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.render_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.cancel_event.clear()
        self.progress.configure(value=0, maximum=cfg.frames)
        self.frame_paths = []
        self.current_preview_path = None
        self.preview_photo = None
        self.preview_label.configure(image="", text="Rendering...")
        self.frame_slider.configure(state=tk.DISABLED, to=1)
        self.vars["frame_slider"].set(1.0)
        self.vars["frame_label"].set(f"Frame 0 / {cfg.frames}")
        self.vars["status"].set("Rendering frames...")

        def run() -> None:
            try:
                video = render_sequence(
                    cfg,
                    lambda done, total, path: self.queue.put(("progress", (done, total, path))),
                    self.cancel_event,
                )
                self.queue.put(("done", video))
            except RenderCancelled:
                self.queue.put(("cancelled", None))
            except Exception as exc:
                self.queue.put(("error", exc))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    done, total, path = payload  # type: ignore[misc]
                    frame_path = Path(path)
                    self.frame_paths.append(frame_path)
                    self.progress.configure(value=done, maximum=total)
                    self.frame_slider.configure(state=tk.NORMAL, to=max(done, 1))
                    self.vars["frame_slider"].set(float(done))
                    self.vars["frame_label"].set(f"Frame {done} / {total}")
                    self._show_frame_path(frame_path)
                    self.vars["status"].set(f"Rendered {done}/{total}: {frame_path.name}")
                elif kind == "done":
                    self.render_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
                    self.vars["status"].set(f"Done: {payload}")
                    messagebox.showinfo("Render complete", f"Video saved:\n{payload}")
                elif kind == "cancelled":
                    self.render_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
                    self.vars["status"].set("Render cancelled")
                elif kind == "error":
                    self.render_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
                    self.vars["status"].set("Render failed")
                    messagebox.showerror("Render failed", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _cancel_render(self) -> None:
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.cancel_button.configure(state=tk.DISABLED)
            self.vars["status"].set("Cancelling after current frame...")

    def _on_slider(self, value: str) -> None:
        if not self.frame_paths:
            return
        index = int(round(float(value))) - 1
        index = max(0, min(index, len(self.frame_paths) - 1))
        self.vars["frame_slider"].set(float(index + 1))
        total = int(float(self.progress.cget("maximum") or len(self.frame_paths)))
        self.vars["frame_label"].set(f"Frame {index + 1} / {total}")
        self._show_frame_path(self.frame_paths[index])

    def _show_frame_path(self, path: Path) -> None:
        self.current_preview_path = path
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if not self.current_preview_path or not self.current_preview_path.exists():
            return
        try:
            image = Image.open(self.current_preview_path).convert("RGB")
            box_width = max(1, self.preview_label.winfo_width())
            box_height = max(1, self.preview_label.winfo_height())
            image.thumbnail((box_width, box_height), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview failed: {exc}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
