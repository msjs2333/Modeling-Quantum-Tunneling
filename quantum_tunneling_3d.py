from __future__ import annotations

# 本脚本完成两件事：
# 1. 用 split-step Fourier 方法演化一维含时薛定谔方程。
# 2. 把复波函数 psi(x,t) 画成三维动画，其中 x 是空间轴，Re/Im 是复平面两轴。
import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# matplotlib 负责绘图和动画输出；numpy 负责复数数组、FFT 和数值积分。
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class SimulationConfig:
    # 空间网格范围。波包从左侧出发，向右撞击位于中心附近的势垒。
    xmin: float = -120.0
    xmax: float = 120.0
    # 网格点数。FFT 对点数较敏感，2 的幂通常比较高效。
    points: int = 2048
    # 每个微小演化步长 dt，以及动画帧数和每帧内部演化步数。
    dt: float = 0.04
    frames: int = 240
    steps_per_frame: int = 8
    # 使用自然单位；默认 hbar=1, mass=1。
    mass: float = 1.0
    hbar: float = 1.0
    # 初始高斯波包参数：中心、宽度、平均波数。
    packet_center: float = -65.0
    packet_sigma: float = 8.0
    packet_k0: float = 1.35
    # 矩形势垒参数。默认能量低于势垒高度，因此会出现隧穿。
    barrier_center: float = 0.0
    barrier_width: float = 3.5
    barrier_height: float = 1.10
    # 边界吸收层参数，用来减少 FFT 周期边界导致的回卷伪影。
    absorber_fraction: float = 0.12
    absorber_strength: float = 0.025


def gaussian_wavepacket(x: np.ndarray, center: float, sigma: float, k0: float) -> np.ndarray:
    # 高斯包络决定波包在空间中的局域范围。
    envelope = np.exp(-((x - center) ** 2) / (4.0 * sigma**2))
    # 平面波相位 exp(i k0 x) 赋予波包向右传播的平均动量。
    phase = np.exp(1j * k0 * x)
    # 归一化因子让连续概率积分接近 1。
    psi = envelope * phase / (2.0 * np.pi * sigma**2) ** 0.25
    return psi


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    # 概率密度为 |psi|^2，离散积分为 sum(|psi|^2) * dx。
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        raise ValueError("Wavefunction norm became zero.")
    # 把波函数缩放到总概率为 1。
    return psi / norm


def rectangular_barrier(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    # 矩形势垒：中心附近宽度为 width 的区域势能为 height，其余为 0。
    half_width = width / 2.0
    return np.where(np.abs(x - center) <= half_width, height, 0.0)


def absorbing_mask(
    x: np.ndarray, fraction: float, strength: float
) -> np.ndarray:
    # 如果吸收层关闭，返回全 1 mask，不改变波函数。
    if fraction <= 0.0 or strength <= 0.0:
        return np.ones_like(x)

    xmin = float(x.min())
    xmax = float(x.max())
    # 吸收层宽度占总空间范围的 fraction。
    edge_width = fraction * (xmax - xmin)
    if edge_width <= 0.0:
        return np.ones_like(x)

    # 中间区域 mask=1；靠近左右边界时逐渐衰减，减少反射回计算区域。
    mask = np.ones_like(x)
    left = x < xmin + edge_width
    right = x > xmax - edge_width
    mask[left] = np.exp(-strength * ((xmin + edge_width - x[left]) / edge_width) ** 2)
    mask[right] = np.exp(-strength * ((x[right] - (xmax - edge_width)) / edge_width) ** 2)
    return mask


def split_step(
    psi: np.ndarray,
    potential: np.ndarray,
    k: np.ndarray,
    cfg: SimulationConfig,
    mask: np.ndarray,
) -> np.ndarray:
    # split-step / Strang splitting：
    # 先在位置空间走半步势能相位，再在动量空间走一步动能相位，最后再走半步势能相位。
    # 这样能把 H = T + V 的指数演化近似为 exp(-iVdt/2) exp(-iTdt) exp(-iVdt/2)。
    potential_half_step = np.exp(-0.5j * potential * cfg.dt / cfg.hbar)
    kinetic_step = np.exp(-0.5j * cfg.hbar * (k**2) * cfg.dt / cfg.mass)

    # V 在位置空间是逐点相乘。
    psi = potential_half_step * psi
    # T 在动量空间是逐点相乘，因此需要 FFT 到 k 空间。
    psi_k = np.fft.fft(psi)
    psi_k *= kinetic_step
    # 再变回位置空间，并完成最后半步势能演化。
    psi = np.fft.ifft(psi_k)
    psi = potential_half_step * psi
    # 吸收边界用于抑制到达边界后的周期回卷。
    return psi * mask


def probabilities(
    x: np.ndarray,
    psi: np.ndarray,
    cfg: SimulationConfig,
    dx: float,
) -> tuple[float, float, float, float]:
    # 分别估算反射区、透射区、势垒附近和全空间的概率。
    prob = np.abs(psi) ** 2
    left_edge = cfg.barrier_center - cfg.barrier_width / 2.0
    right_edge = cfg.barrier_center + cfg.barrier_width / 2.0
    # margin 避免把还停留在势垒附近的概率误判为反射或透射。
    margin = 2.0 * cfg.packet_sigma

    reflected = float(np.sum(prob[x < left_edge - margin]) * dx)
    transmitted = float(np.sum(prob[x > right_edge + margin]) * dx)
    near_barrier = float(np.sum(prob[(x >= left_edge - margin) & (x <= right_edge + margin)]) * dx)
    total = float(np.sum(prob) * dx)
    return reflected, transmitted, near_barrier, total


def simulate(cfg: SimulationConfig) -> dict[str, np.ndarray]:
    # 建立等间距空间网格；endpoint=False 与 FFT 的周期网格约定更一致。
    x = np.linspace(cfg.xmin, cfg.xmax, cfg.points, endpoint=False)
    dx = float(x[1] - x[0])
    # FFT 频率换算为波数 k，用于动能相位 exp(-i hbar k^2 dt / 2m)。
    k = 2.0 * np.pi * np.fft.fftfreq(cfg.points, d=dx)
    # 生成势垒和边界吸收 mask。
    potential = rectangular_barrier(x, cfg.barrier_center, cfg.barrier_width, cfg.barrier_height)
    mask = absorbing_mask(x, cfg.absorber_fraction, cfg.absorber_strength)

    # 初始化并归一化波函数。
    psi = gaussian_wavepacket(x, cfg.packet_center, cfg.packet_sigma, cfg.packet_k0)
    psi = normalize(psi, dx)

    # snapshots 保存每一帧的复波函数；diagnostics 保存 R/T/near/norm。
    snapshots = np.empty((cfg.frames, cfg.points), dtype=np.complex128)
    diagnostics = np.empty((cfg.frames, 4), dtype=float)
    times = np.empty(cfg.frames, dtype=float)

    for frame in range(cfg.frames):
        # 先记录当前帧，再向前演化到下一帧。
        snapshots[frame] = psi
        diagnostics[frame] = probabilities(x, psi, cfg, dx)
        times[frame] = frame * cfg.steps_per_frame * cfg.dt

        # 每个动画帧之间执行多个小时间步，使动画不必保存过多帧。
        for _ in range(cfg.steps_per_frame):
            psi = split_step(psi, potential, k, cfg, mask)

    # 用字典返回，方便绘图函数按名称取数据。
    return {
        "x": x,
        "potential": potential,
        "snapshots": snapshots,
        "diagnostics": diagnostics,
        "times": times,
        "dx": np.array(dx),
    }


def add_barrier(ax, cfg: SimulationConfig, amp_limit: float) -> None:
    # 用一个半透明长方体表示矩形势垒在复平面中的位置。
    left = cfg.barrier_center - cfg.barrier_width / 2.0
    right = cfg.barrier_center + cfg.barrier_width / 2.0
    y0, y1 = -amp_limit, amp_limit
    z0, z1 = -amp_limit, amp_limit

    # 这里只画四个侧面，留出顶部/底部，避免遮挡波函数曲线。
    faces = [
        [(left, y0, z0), (right, y0, z0), (right, y1, z0), (left, y1, z0)],
        [(left, y0, z1), (right, y0, z1), (right, y1, z1), (left, y1, z1)],
        [(left, y0, z0), (left, y0, z1), (left, y1, z1), (left, y1, z0)],
        [(right, y0, z0), (right, y0, z1), (right, y1, z1), (right, y1, z0)],
    ]
    wall = Poly3DCollection(
        faces,
        facecolors=(0.93, 0.34, 0.18, 0.16),
        edgecolors=(0.93, 0.34, 0.18, 0.55),
        linewidths=0.7,
    )
    # 把势垒几何体加入 3D 坐标轴。
    ax.add_collection3d(wall)


def ffmpeg_candidates(explicit_path: str | None) -> list[str]:
    # 按优先级收集可能的 ffmpeg 路径。这里不用只依赖 Python 的 PATH，
    # 因为 IDE/Conda/Python 进程拿到的 PATH 可能和 Windows Terminal 不一致。
    candidates: list[str] = []

    def add(value: str | os.PathLike[str] | None) -> None:
        if not value:
            return
        text = str(value).strip().strip('"')
        if text and text not in candidates:
            candidates.append(text)
        # WinGet 的 Links 目录通常放的是符号链接。把真实目标也加入候选列表，
        # 避免 Matplotlib/Python 对 reparse point 的处理和普通终端不同。
        try:
            resolved = str(Path(text).resolve(strict=False))
        except OSError:
            return
        if resolved and resolved != text and resolved not in candidates:
            candidates.append(resolved)

    add(explicit_path)
    add(os.environ.get("FFMPEG_BINARY"))
    add(os.environ.get("IMAGEIO_FFMPEG_EXE"))
    add(shutil.which("ffmpeg"))

    # 有些用户在 PowerShell profile 里追加 PATH，Python/IDE 进程本身看不到。
    # 这里额外询问 PowerShell：如果终端里 `ffmpeg` 能运行，通常能拿到 Source。
    try:
        ps_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-Command ffmpeg -ErrorAction SilentlyContinue).Source",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        for line in ps_result.stdout.splitlines():
            add(line)
    except (OSError, subprocess.SubprocessError):
        pass

    local_appdata = os.environ.get("LOCALAPPDATA")
    userprofile = os.environ.get("USERPROFILE")
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    program_data = os.environ.get("ProgramData")

    if local_appdata:
        add(Path(local_appdata) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
        add(Path(local_appdata) / "Microsoft" / "WindowsApps" / "ffmpeg.exe")
        winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        try:
            for path in winget_packages.glob("**/ffmpeg.exe"):
                add(path)
        except OSError:
            # 某些 WinGet 包目录可能有 ACL 限制；跳过不可遍历目录。
            pass

    if userprofile:
        add(Path(userprofile) / "scoop" / "shims" / "ffmpeg.exe")
        add(Path(userprofile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe")
    if program_data:
        add(Path(program_data) / "chocolatey" / "bin" / "ffmpeg.exe")
    if program_files:
        add(Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe")
    if program_files_x86:
        add(Path(program_files_x86) / "ffmpeg" / "bin" / "ffmpeg.exe")

    add(Path("C:/ffmpeg/bin/ffmpeg.exe"))
    add("ffmpeg")
    return candidates


def resolve_ffmpeg(explicit_path: str | None) -> str:
    # 逐个候选路径执行 `ffmpeg -version`，只有真的能运行才交给 Matplotlib。
    errors: list[str] = []
    for candidate in ffmpeg_candidates(explicit_path):
        try:
            result = subprocess.run(
                [candidate, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        if result.returncode == 0:
            return candidate
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[0] if stderr else f"exit code {result.returncode}"
        errors.append(f"{candidate}: {detail}")

    message = (
        "Cannot find a runnable ffmpeg. Run `where ffmpeg` in the terminal that works, "
        "then pass that exact path with --ffmpeg-path. Tried:\n  "
        + "\n  ".join(errors[:12])
    )
    raise RuntimeError(message)


def make_animation(
    data: dict[str, np.ndarray],
    cfg: SimulationConfig,
    output: Path,
    fps: int,
    dpi: int,
    pix_fmt: str,
    ffmpeg_path: str | None,
) -> None:
    # 从模拟结果中取出绘图需要的数据。
    x = data["x"]
    potential = data["potential"]
    snapshots = data["snapshots"]
    diagnostics = data["diagnostics"]
    times = data["times"]

    # amp_limit 决定 Re/Im 两个方向的可视范围。
    amp_limit = max(0.32, float(np.max(np.abs(snapshots))) * 1.25)
    # 概率密度 |psi|^2 的量纲和波函数幅值不同，这里缩放到图中底面显示。
    prob_scale = 0.65 * amp_limit / float(np.max(np.abs(snapshots[0]) ** 2))

    plt.style.use("dark_background")
    # 画布尺寸。这里只保留上一版默认值；如需放大主图，可手动调大 figsize。
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    # 子图边距。top 控制标题上方空间；值越小，顶部留白越多。
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.88)

    # 设置三维坐标轴范围。
    ax.set_xlim(cfg.xmin, cfg.xmax)
    # y/z 轴范围决定 Re(psi)、Im(psi) 的显示高度。
    # 缩小范围会让波形看起来更大，但过小会裁切波峰。
    ax.set_ylim(-amp_limit, amp_limit)
    ax.set_zlim(-amp_limit, amp_limit)
    # 坐标轴文字和刻度字号。
    ax.set_xlabel("x", fontsize=14, labelpad=12)
    ax.set_ylabel("Re(psi)", fontsize=14, labelpad=12)
    ax.set_zlabel("Im(psi)", fontsize=14, labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    # 标题设置。如果标题超出画布，可调小 fontsize/pad 或降低 subplots_adjust(top=...)。
    ax.set_title(
        "Quantum tunneling: wavefunction in the complex plane",
        fontsize=24,
        pad=28,
    )
    # 三维坐标盒比例。第一个数越大，x 轴越长；后两个数控制复平面大小。
    ax.set_box_aspect((3.2, 1.2, 1.2))
    # 固定视角；不在 update() 中改变 view_init，因此动画视角保持恒定。
    ax.view_init(elev=24, azim=-58)

    # 灰色中心线表示 Re=0, Im=0 的 x 轴。
    ax.plot([cfg.xmin, cfg.xmax], [0, 0], [0, 0], color="0.5", linewidth=0.8, alpha=0.75)
    add_barrier(ax, cfg, amp_limit)

    # 在底面画势垒位置提示线，方便读出势垒宽度。
    barrier_profile = np.where(potential > 0, -0.92 * amp_limit, np.nan)
    ax.plot(
        x,
        barrier_profile,
        np.full_like(x, -0.92 * amp_limit),
        color="#ff8a47",
        linewidth=4.0,
        alpha=0.95,
    )

    # 取第一帧初始化所有曲线对象；后续 update() 只更新这些对象的数据。
    psi0 = snapshots[0]
    # 主曲线：真正的复波函数轨迹 (x, Re(psi), Im(psi))。
    wave_line, = ax.plot(
        x,
        psi0.real,
        psi0.imag,
        color="#e879f9",
        linewidth=2.6,
        label="psi in complex plane",
    )
    # 实部投影：把 Im 固定到底面，显示 x-Re(psi) 平面中的变化。
    re_projection, = ax.plot(
        x,
        psi0.real,
        np.full_like(x, -amp_limit),
        color="#ff5e5e",
        linewidth=1.5,
        alpha=0.92,
        label="Re projection",
    )
    # 虚部投影：把 Re 固定到后侧平面，即 Re(psi)=+amp_limit。
    # 如果要放回靠前平面，把这里和 update() 中的 amp_limit 改成 -amp_limit。
    im_projection, = ax.plot(
        x,
        np.full_like(x, amp_limit),
        psi0.imag,
        color="#5ea8ff",
        linewidth=1.5,
        alpha=0.92,
        label="Im projection (back plane)",
    )
    # 概率密度投影：在底面用白线显示 |psi|^2 的包络。
    prob_line, = ax.plot(
        x,
        np.abs(psi0) ** 2 * prob_scale - amp_limit,
        np.full_like(x, -amp_limit),
        color="#e8e8e8",
        linewidth=1.2,
        alpha=0.75,
    )
    # 图例位置用 loc 和 bbox_to_anchor 控制；如果与文字或波形重叠，在这里移动。
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        facecolor=(0.02, 0.02, 0.02, 0.62),
        edgecolor="0.35",
        fontsize=11,
    )
    # 左上角诊断文字：显示时间、反射概率、透射概率、势垒附近概率和总范数。
    text = ax.text2D(0.02, 0.91, "", transform=ax.transAxes, color="white", fontsize=13)

    def update(frame: int):
        # 取当前帧波函数，并更新主曲线与三个投影。
        psi = snapshots[frame]
        wave_line.set_data_3d(x, psi.real, psi.imag)
        re_projection.set_data_3d(x, psi.real, np.full_like(x, -amp_limit))
        im_projection.set_data_3d(x, np.full_like(x, amp_limit), psi.imag)
        # 更新概率密度包络。
        prob_line.set_data_3d(
            x,
            np.abs(psi) ** 2 * prob_scale - amp_limit,
            np.full_like(x, -amp_limit),
        )
        # 更新每帧的诊断数值。
        reflected, transmitted, near_barrier, total = diagnostics[frame]
        text.set_text(
            f"t = {times[frame]:5.2f}   "
            f"R = {reflected:5.3f}   T = {transmitted:5.3f}   "
            f"near barrier = {near_barrier:5.3f}   norm = {total:5.3f}"
        )
        return wave_line, re_projection, im_projection, prob_line, text

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    # 根据输出文件后缀选择动画 writer。
    if suffix == ".mp4":
        # 不写死 WinGet 包目录。优先使用命令行传入路径，再搜索 PATH 和常见 Windows 安装位置。
        # yuv420p 兼容性最好；yuv444p 色彩采样更完整，但部分播放器/演示软件支持较差。
        resolved_ffmpeg = resolve_ffmpeg(ffmpeg_path)
        matplotlib.rcParams["animation.ffmpeg_path"] = resolved_ffmpeg
        writer = FFMpegWriter(
            fps=fps,
            bitrate=3600,
            codec="libx264",
            extra_args=["-pix_fmt", pix_fmt],
        )
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("Output must end in .gif or .mp4")

    # FuncAnimation 负责逐帧调用 update()。
    anim = FuncAnimation(fig, update, frames=cfg.frames, interval=1000 / fps, blit=False)

    # 保存动画并关闭 figure，避免脚本结束前占用内存。
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    # 命令行参数用于快速改变物理参数和输出格式，不需要改源码。
    parser = argparse.ArgumentParser(
        description="3D complex-plane animation of 1D quantum tunneling."
    )
    # 输出路径：后缀为 .gif 时用 PillowWriter，后缀为 .mp4 时用 ffmpeg。
    parser.add_argument("--output", default="output/tunneling_3d.mp4", help="Path to .gif or .mp4")
    # 数值演化和动画采样参数。
    parser.add_argument("--frames", type=int, default=720)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--points", type=int, default=2048)
    # 初始波包参数。
    parser.add_argument("--k0", type=float, default=1.35, help="Initial wave number")
    parser.add_argument("--sigma", type=float, default=8.0, help="Initial Gaussian width")
    # 势垒参数。
    parser.add_argument("--barrier-height", type=float, default=1.10)
    parser.add_argument("--barrier-width", type=float, default=3.5)
    # 动画输出参数。
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--pix-fmt",
        default="yuv420p",
        choices=["yuv420p", "yuv444p"],
        help="MP4 pixel format. yuv420p is broadly compatible; yuv444p keeps full chroma.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Optional explicit path to ffmpeg.exe. If omitted, ffmpeg is resolved from PATH.",
    )
    return parser.parse_args()


def main() -> None:
    # 读取命令行参数，并覆盖默认配置中允许从命令行调整的部分。
    args = parse_args()
    cfg = SimulationConfig(
        dt=args.dt,
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        points=args.points,
        packet_k0=args.k0,
        packet_sigma=args.sigma,
        barrier_height=args.barrier_height,
        barrier_width=args.barrier_width,
    )

    # 先完成数值模拟，再计算最终帧的诊断量。
    data = simulate(cfg)
    final_r, final_t, final_near, final_norm = data["diagnostics"][-1]
    energy = cfg.hbar**2 * cfg.packet_k0**2 / (2.0 * cfg.mass)
    print(f"Initial packet energy: {energy:.4f}")
    print(f"Barrier height:        {cfg.barrier_height:.4f}")
    print(f"Final reflected R:     {final_r:.4f}")
    print(f"Final transmitted T:   {final_t:.4f}")
    print(f"Final near barrier:    {final_near:.4f}")
    print(f"Final norm:            {final_norm:.4f}")

    # 按指定格式保存动画。
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


if __name__ == "__main__":
    # 直接运行脚本时进入主流程；被其他脚本 import 时不会自动生成动画。
    main()
