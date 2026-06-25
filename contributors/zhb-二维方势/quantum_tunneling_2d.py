from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REAL_CMAP = LinearSegmentedColormap.from_list(
    "real_wave_dark",
    ["#050816", "#0b1f3a", "#1f9cf0", "#d8fbff"],
)
IMAG_CMAP = LinearSegmentedColormap.from_list(
    "imag_wave_dark",
    ["#070512", "#26123f", "#d946ef", "#ffe4ff"],
)
OUTPUT_DIR = Path("output")


@dataclass(frozen=True)
class SimulationConfig:
    xmin: float = -100.0
    xmax: float = 100.0
    ymin: float = -60.0
    ymax: float = 60.0
    nx: int = 256
    ny: int = 192
    dt: float = 0.035
    frames: int = 360
    steps_per_frame: int = 5
    mass: float = 1.0
    hbar: float = 1.0
    packet_x0: float = -55.0
    packet_y0: float = 0.0
    packet_sigma_x: float = 8.0
    packet_sigma_y: float = 10.0
    packet_kx0: float = 1.42
    packet_ky0: float = 0.0
    barrier_x0: float = 0.0
    barrier_y0: float = 0.0
    barrier_x_width: float = 18.0
    barrier_y_width: float = 36.0
    barrier_height: float = 0.95
    absorber_fraction: float = 0.12
    absorber_strength: float = 0.045


def gaussian_wavepacket_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    kx0: float,
    ky0: float,
) -> np.ndarray:
    envelope = np.exp(
        -((x_grid - x0) ** 2) / (4.0 * sigma_x**2)
        -((y_grid - y0) ** 2) / (4.0 * sigma_y**2)
    )
    phase = np.exp(1j * (kx0 * x_grid + ky0 * y_grid))
    return envelope * phase


def normalize_2d(psi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy)
    if norm == 0:
        raise ValueError("Wavefunction norm became zero.")
    return psi / norm


def rectangular_barrier_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
    width_x: float,
    width_y: float,
    height: float,
) -> np.ndarray:
    inside = (
        (np.abs(x_grid - x0) <= width_x / 2.0)
        & (np.abs(y_grid - y0) <= width_y / 2.0)
    )
    return np.where(inside, height, 0.0)


def absorbing_mask_1d(axis: np.ndarray, fraction: float, strength: float) -> np.ndarray:
    if fraction <= 0.0 or strength <= 0.0:
        return np.ones_like(axis)

    axis_min = float(axis.min())
    axis_max = float(axis.max())
    edge_width = fraction * (axis_max - axis_min)
    if edge_width <= 0.0:
        return np.ones_like(axis)

    mask = np.ones_like(axis)
    left = axis < axis_min + edge_width
    right = axis > axis_max - edge_width
    mask[left] = np.exp(-strength * ((axis_min + edge_width - axis[left]) / edge_width) ** 2)
    mask[right] = np.exp(-strength * ((axis[right] - (axis_max - edge_width)) / edge_width) ** 2)
    return mask


def absorbing_mask_2d(x: np.ndarray, y: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    mask_x = absorbing_mask_1d(x, cfg.absorber_fraction, cfg.absorber_strength)
    mask_y = absorbing_mask_1d(y, cfg.absorber_fraction, cfg.absorber_strength)
    return mask_y[:, None] * mask_x[None, :]


def split_step_2d(
    psi: np.ndarray,
    potential: np.ndarray,
    k_squared: np.ndarray,
    cfg: SimulationConfig,
    mask: np.ndarray,
) -> np.ndarray:
    potential_half_step = np.exp(-0.5j * potential * cfg.dt / cfg.hbar)
    kinetic_step = np.exp(-0.5j * cfg.hbar * k_squared * cfg.dt / cfg.mass)

    psi = potential_half_step * psi
    psi_k = np.fft.fft2(psi)
    psi_k *= kinetic_step
    psi = np.fft.ifft2(psi_k)
    psi = potential_half_step * psi
    return psi * mask


def probabilities_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    psi: np.ndarray,
    cfg: SimulationConfig,
    dx: float,
    dy: float,
) -> tuple[float, float, float, float, float]:
    density = np.abs(psi) ** 2
    left_edge = cfg.barrier_x0 - cfg.barrier_x_width / 2.0
    right_edge = cfg.barrier_x0 + cfg.barrier_x_width / 2.0
    bottom_edge = cfg.barrier_y0 - cfg.barrier_y_width / 2.0
    top_edge = cfg.barrier_y0 + cfg.barrier_y_width / 2.0
    margin_x = 2.0 * cfg.packet_sigma_x
    margin_y = 1.0 * cfg.packet_sigma_y
    area = dx * dy

    reflected = float(np.sum(density[x_grid < left_edge - margin_x]) * area)
    transmitted = float(np.sum(density[x_grid > right_edge + margin_x]) * area)
    bypass = float(
        np.sum(
            density[
                (np.abs(x_grid - cfg.barrier_x0) <= margin_x + cfg.barrier_x_width / 2.0)
                & ((y_grid < bottom_edge - margin_y) | (y_grid > top_edge + margin_y))
            ]
        )
        * area
    )
    near_barrier = float(
        np.sum(
            density[
                (x_grid >= left_edge - margin_x)
                & (x_grid <= right_edge + margin_x)
                & (y_grid >= bottom_edge - margin_y)
                & (y_grid <= top_edge + margin_y)
            ]
        )
        * area
    )
    total = float(np.sum(density) * area)
    return reflected, transmitted, bypass, near_barrier, total


def simulate(cfg: SimulationConfig) -> dict[str, np.ndarray]:
    x = np.linspace(cfg.xmin, cfg.xmax, cfg.nx, endpoint=False)
    y = np.linspace(cfg.ymin, cfg.ymax, cfg.ny, endpoint=False)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    kx = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(cfg.ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k_squared = kx_grid**2 + ky_grid**2

    potential = rectangular_barrier_2d(
        x_grid,
        y_grid,
        cfg.barrier_x0,
        cfg.barrier_y0,
        cfg.barrier_x_width,
        cfg.barrier_y_width,
        cfg.barrier_height,
    )
    mask = absorbing_mask_2d(x, y, cfg)

    psi = gaussian_wavepacket_2d(
        x_grid,
        y_grid,
        cfg.packet_x0,
        cfg.packet_y0,
        cfg.packet_sigma_x,
        cfg.packet_sigma_y,
        cfg.packet_kx0,
        cfg.packet_ky0,
    )
    psi = normalize_2d(psi, dx, dy)

    density_snapshots = np.empty((cfg.frames, cfg.ny, cfg.nx), dtype=np.float32)
    wave_snapshots = np.empty((cfg.frames, cfg.ny, cfg.nx), dtype=np.complex64)
    diagnostics = np.empty((cfg.frames, 5), dtype=float)
    times = np.empty(cfg.frames, dtype=float)

    for frame in range(cfg.frames):
        wave_snapshots[frame] = psi.astype(np.complex64)
        density_snapshots[frame] = np.abs(psi) ** 2
        diagnostics[frame] = probabilities_2d(x_grid, y_grid, psi, cfg, dx, dy)
        times[frame] = frame * cfg.steps_per_frame * cfg.dt

        for _ in range(cfg.steps_per_frame):
            psi = split_step_2d(psi, potential, k_squared, cfg, mask)

    return {
        "x": x,
        "y": y,
        "potential": potential,
        "wavefunction": wave_snapshots,
        "density": density_snapshots,
        "diagnostics": diagnostics,
        "times": times,
        "dx": np.array(dx),
        "dy": np.array(dy),
    }


def save_animation_file(
    animation: FuncAnimation,
    output: Path,
    fps: int,
    dpi: int,
    pix_fmt: str,
    ffmpeg_path: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        if ffmpeg_path:
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        writer = FFMpegWriter(
            fps=fps,
            bitrate=4200,
            codec="libx264",
            extra_args=["-pix_fmt", pix_fmt],
        )
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("Output must end in .mp4 or .gif")

    animation.save(output, writer=writer, dpi=dpi)


def make_animation(
    data: dict[str, np.ndarray],
    cfg: SimulationConfig,
    output: Path,
    fps: int,
    dpi: int,
    pix_fmt: str,
    ffmpeg_path: str | None,
) -> None:
    density = data["density"]
    wavefunction = data["wavefunction"]
    diagnostics = data["diagnostics"]
    times = data["times"]

    vmax = float(np.percentile(density, 99.7))
    if vmax <= 0:
        vmax = float(np.max(density))
    amplitude_limit = float(np.percentile(np.abs(wavefunction), 98.0))
    if amplitude_limit <= 0:
        amplitude_limit = float(np.max(np.abs(wavefunction)))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.8), constrained_layout=True)

    barrier_left = cfg.barrier_x0 - cfg.barrier_x_width / 2.0
    barrier_bottom = cfg.barrier_y0 - cfg.barrier_y_width / 2.0

    density_image = axes[0].imshow(
        density[0],
        extent=[cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax],
        origin="lower",
        cmap="magma",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax),
        interpolation="bilinear",
        animated=True,
    )
    real_image = axes[1].imshow(
        np.abs(wavefunction[0].real),
        extent=[cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax],
        origin="lower",
        cmap=REAL_CMAP,
        norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=amplitude_limit),
        interpolation="bilinear",
        animated=True,
    )
    imag_image = axes[2].imshow(
        np.abs(wavefunction[0].imag),
        extent=[cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax],
        origin="lower",
        cmap=IMAG_CMAP,
        norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=amplitude_limit),
        interpolation="bilinear",
        animated=True,
    )
    for ax, title in zip(axes, (r"Probability density $|\psi|^2$", r"Real-part wave $|$Re($\psi$)$|$", r"Imaginary-part wave $|$Im($\psi$)$|$")):
        ax.add_patch(
            Rectangle(
                (barrier_left, barrier_bottom),
                cfg.barrier_x_width,
                cfg.barrier_y_width,
                fill=False,
                edgecolor="#61dafb",
                linewidth=2.0,
            )
        )
        ax.set_facecolor("#050816")
        ax.set_xlim(cfg.xmin, cfg.xmax)
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=12)

    density_colorbar = fig.colorbar(density_image, ax=axes[0], fraction=0.046, pad=0.025)
    density_colorbar.set_label(r"$|\psi(x,y,t)|^2$")
    real_colorbar = fig.colorbar(real_image, ax=axes[1], fraction=0.046, pad=0.025)
    real_colorbar.set_label(r"$|\mathrm{Re}(\psi)|$")
    imag_colorbar = fig.colorbar(imag_image, ax=axes[2], fraction=0.046, pad=0.025)
    imag_colorbar.set_label(r"$|\mathrm{Im}(\psi)|$")

    text = fig.text(0.5, 0.985, "", ha="center", va="top", color="white", fontsize=12)

    def update(frame: int):
        psi = wavefunction[frame]
        density_image.set_array(density[frame])
        real_image.set_array(np.abs(psi.real))
        imag_image.set_array(np.abs(psi.imag))
        reflected, transmitted, bypass, near_barrier, total = diagnostics[frame]
        text.set_text(
            f"t={times[frame]:5.2f}   "
            f"R={reflected:5.3f}   T={transmitted:5.3f}   "
            f"bypass={bypass:5.3f}   near={near_barrier:5.3f}   norm={total:5.3f}"
        )
        return density_image, real_image, imag_image, text

    animation = FuncAnimation(fig, update, frames=cfg.frames, interval=1000 / fps, blit=False)
    save_animation_file(animation, output, fps=fps, dpi=dpi, pix_fmt=pix_fmt, ffmpeg_path=ffmpeg_path)
    plt.close(fig)


def make_component_animation(
    data: dict[str, np.ndarray],
    cfg: SimulationConfig,
    output: Path,
    fps: int,
    dpi: int,
    pix_fmt: str,
    ffmpeg_path: str | None,
) -> None:
    wavefunction = data["wavefunction"]
    times = data["times"]

    amplitude_limit = float(np.percentile(np.abs(wavefunction), 98.0))
    if amplitude_limit <= 0:
        amplitude_limit = float(np.max(np.abs(wavefunction)))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)

    barrier_left = cfg.barrier_x0 - cfg.barrier_x_width / 2.0
    barrier_bottom = cfg.barrier_y0 - cfg.barrier_y_width / 2.0

    images = []
    titles = (r"Real-part wave $|\mathrm{Re}(\psi)|$", r"Imaginary-part wave $|\mathrm{Im}(\psi)|$")
    initial_components = (np.abs(wavefunction[0].real), np.abs(wavefunction[0].imag))
    cmaps = (REAL_CMAP, IMAG_CMAP)
    for ax, title, component, cmap in zip(axes, titles, initial_components, cmaps):
        image = ax.imshow(
            component,
            extent=[cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax],
            origin="lower",
            cmap=cmap,
            norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=amplitude_limit),
            interpolation="bilinear",
            animated=True,
        )
        ax.add_patch(
            Rectangle(
                (barrier_left, barrier_bottom),
                cfg.barrier_x_width,
                cfg.barrier_y_width,
                fill=False,
                edgecolor="#61dafb",
                linewidth=2.0,
            )
        )
        ax.set_facecolor("#050816")
        ax.set_title(title, fontsize=13)
        ax.set_xlim(cfg.xmin, cfg.xmax)
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        images.append(image)

    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.036, pad=0.025)
    colorbar.set_label("wavefunction component amplitude")
    text = fig.text(0.5, 0.985, "", ha="center", va="top", color="white", fontsize=12)

    def update(frame: int):
        psi = wavefunction[frame]
        images[0].set_array(np.abs(psi.real))
        images[1].set_array(np.abs(psi.imag))
        text.set_text(f"Real and imaginary part amplitudes through the rectangular barrier   t={times[frame]:5.2f}")
        return images[0], images[1], text

    animation = FuncAnimation(fig, update, frames=cfg.frames, interval=1000 / fps, blit=False)
    save_animation_file(animation, output, fps=fps, dpi=dpi, pix_fmt=pix_fmt, ffmpeg_path=ffmpeg_path)
    plt.close(fig)


def add_barrier_box_3d(ax, cfg: SimulationConfig, z_height: float) -> None:
    left = cfg.barrier_x0 - cfg.barrier_x_width / 2.0
    right = cfg.barrier_x0 + cfg.barrier_x_width / 2.0
    bottom = cfg.barrier_y0 - cfg.barrier_y_width / 2.0
    top = cfg.barrier_y0 + cfg.barrier_y_width / 2.0
    z0 = 0.0
    z1 = z_height

    faces = [
        [(left, bottom, z0), (right, bottom, z0), (right, top, z0), (left, top, z0)],
        [(left, bottom, z1), (right, bottom, z1), (right, top, z1), (left, top, z1)],
        [(left, bottom, z0), (right, bottom, z0), (right, bottom, z1), (left, bottom, z1)],
        [(left, top, z0), (right, top, z0), (right, top, z1), (left, top, z1)],
        [(left, bottom, z0), (left, top, z0), (left, top, z1), (left, bottom, z1)],
        [(right, bottom, z0), (right, top, z0), (right, top, z1), (right, bottom, z1)],
    ]
    box = Poly3DCollection(
        faces,
        facecolors=(0.38, 0.85, 0.98, 0.18),
        edgecolors=(0.38, 0.85, 0.98, 0.75),
        linewidths=0.7,
    )
    ax.add_collection3d(box)


def make_3d_animation(
    data: dict[str, np.ndarray],
    cfg: SimulationConfig,
    output: Path,
    fps: int,
    dpi: int,
    frame_step: int,
    grid_step: int,
    pix_fmt: str,
    ffmpeg_path: str | None,
) -> None:
    x = data["x"][::grid_step]
    y = data["y"][::grid_step]
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    density = data["density"][:, ::grid_step, ::grid_step]
    wavefunction = data["wavefunction"][:, ::grid_step, ::grid_step]
    diagnostics = data["diagnostics"]
    times = data["times"]

    density_limit = float(np.percentile(density, 99.8))
    if density_limit <= 0:
        density_limit = float(np.max(density))
    density_limit *= 1.25
    component_limit = float(np.percentile(np.abs(wavefunction), 98.0))
    if component_limit <= 0:
        component_limit = float(np.max(np.abs(wavefunction)))
    component_limit *= 0.95

    frames = np.arange(0, cfg.frames, max(1, frame_step), dtype=int)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16.4, 6.2))
    axes = [fig.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.86, wspace=0.02)

    panel_specs = [
        {
            "title": r"Probability density $|\psi|^2$",
            "zlabel": r"$|\psi|^2$",
            "zlim": (0.0, density_limit),
            "barrier_height": density_limit * 0.45,
            "cmap": "magma",
            "vmin": 0.0,
            "vmax": density_limit / 1.25,
        },
        {
            "title": r"Real-part wave $|\mathrm{Re}(\psi)|$",
            "zlabel": r"$|\mathrm{Re}(\psi)|$",
            "zlim": (0.0, component_limit),
            "barrier_height": component_limit * 0.35,
            "cmap": REAL_CMAP,
            "vmin": 0.0,
            "vmax": component_limit,
        },
        {
            "title": r"Imaginary-part wave $|\mathrm{Im}(\psi)|$",
            "zlabel": r"$|\mathrm{Im}(\psi)|$",
            "zlim": (0.0, component_limit),
            "barrier_height": component_limit * 0.35,
            "cmap": IMAG_CMAP,
            "vmin": 0.0,
            "vmax": component_limit,
        },
    ]

    initial_surfaces = [density[0], np.abs(wavefunction[0].real), np.abs(wavefunction[0].imag)]
    surfaces = []
    for ax, spec, initial_surface in zip(axes, panel_specs, initial_surfaces):
        ax.set_xlim(cfg.xmin, cfg.xmax)
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_zlim(*spec["zlim"])
        ax.set_facecolor("#050816")
        ax.set_xlabel("x", labelpad=7)
        ax.set_ylabel("y", labelpad=7)
        ax.set_zlabel(spec["zlabel"], labelpad=7)
        ax.set_title(spec["title"], fontsize=11, pad=12)
        ax.view_init(elev=32, azim=-62)
        ax.set_box_aspect((2.4, 1.6, 0.72))
        add_barrier_box_3d(ax, cfg, spec["barrier_height"])
        surfaces.append(
            ax.plot_surface(
                x_grid,
                y_grid,
                initial_surface,
                cmap=spec["cmap"],
                vmin=spec["vmin"],
                vmax=spec["vmax"],
                linewidth=0,
                antialiased=True,
                shade=True,
            )
        )

    text = fig.text(0.5, 0.975, "", ha="center", va="top", color="white", fontsize=12)

    def update(frame_number: int):
        frame = int(frames[frame_number])
        current_surfaces = [density[frame], np.abs(wavefunction[frame].real), np.abs(wavefunction[frame].imag)]
        for index, (ax, spec, current_surface) in enumerate(zip(axes, panel_specs, current_surfaces)):
            surfaces[index].remove()
            surfaces[index] = ax.plot_surface(
                x_grid,
                y_grid,
                current_surface,
                cmap=spec["cmap"],
                vmin=spec["vmin"],
                vmax=spec["vmax"],
                linewidth=0,
                antialiased=True,
                shade=True,
            )
        reflected, transmitted, bypass, near_barrier, total = diagnostics[frame]
        text.set_text(
            f"t={times[frame]:5.2f}   R={reflected:5.3f}   T={transmitted:5.3f}   "
            f"bypass={bypass:5.3f}   near={near_barrier:5.3f}   norm={total:5.3f}"
        )
        return (*surfaces, text)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    save_animation_file(animation, output, fps=fps, dpi=dpi, pix_fmt=pix_fmt, ffmpeg_path=ffmpeg_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D split-step Fourier simulation of quantum tunneling.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "tunneling_2d.mp4"), help="Path to the 2D .mp4 or .gif")
    parser.add_argument(
        "--component-output",
        default=str(OUTPUT_DIR / "tunneling_2d_re_im.mp4"),
        help="Path to the Re/Im component .mp4 or .gif",
    )
    parser.add_argument("--output-3d", default=str(OUTPUT_DIR / "tunneling_2d_3d.mp4"), help="Path to the 3D .mp4 or .gif")
    parser.add_argument("--three-d-frame-step", type=int, default=3)
    parser.add_argument("--three-d-grid-step", type=int, default=4)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.035)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=192)
    parser.add_argument("--kx0", type=float, default=1.42)
    parser.add_argument("--ky0", type=float, default=0.0)
    parser.add_argument("--sigma-x", type=float, default=8.0)
    parser.add_argument("--sigma-y", type=float, default=10.0)
    parser.add_argument("--barrier-height", type=float, default=0.95)
    parser.add_argument("--barrier-x-width", type=float, default=18.0)
    parser.add_argument("--barrier-y-width", type=float, default=36.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        choices=["yuv420p", "yuv444p"],
        help="MP4 pixel format. yuv420p is most compatible.",
    )
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit path to ffmpeg.exe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ffmpeg_path:
        ffmpeg_dir = str(Path(args.ffmpeg_path).resolve().parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    cfg = SimulationConfig(
        dt=args.dt,
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        nx=args.nx,
        ny=args.ny,
        packet_kx0=args.kx0,
        packet_ky0=args.ky0,
        packet_sigma_x=args.sigma_x,
        packet_sigma_y=args.sigma_y,
        barrier_height=args.barrier_height,
        barrier_x_width=args.barrier_x_width,
        barrier_y_width=args.barrier_y_width,
    )

    data = simulate(cfg)
    final_r, final_t, final_bypass, final_near, final_norm = data["diagnostics"][-1]
    energy = cfg.hbar**2 * (cfg.packet_kx0**2 + cfg.packet_ky0**2) / (2.0 * cfg.mass)

    print(f"Initial packet energy: {energy:.4f}")
    print(f"Barrier height:        {cfg.barrier_height:.4f}")
    print(f"Final reflected R:     {final_r:.4f}")
    print(f"Final transmitted T:   {final_t:.4f}")
    print(f"Final bypass:          {final_bypass:.4f}")
    print(f"Final near barrier:    {final_near:.4f}")
    print(f"Final norm:            {final_norm:.4f}")

    make_animation(
        data,
        cfg,
        Path(args.output),
        fps=args.fps,
        dpi=args.dpi,
        pix_fmt=args.pix_fmt,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(f"Wrote {args.output}")
    make_3d_animation(
        data,
        cfg,
        Path(args.output_3d),
        fps=args.fps,
        dpi=args.dpi,
        frame_step=args.three_d_frame_step,
        grid_step=args.three_d_grid_step,
        pix_fmt=args.pix_fmt,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(f"Wrote {args.output_3d}")


if __name__ == "__main__":
    main()
