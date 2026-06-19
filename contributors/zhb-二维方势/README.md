
# zhb - 二维方势

本目录保存 zhb 负责的二维方势量子波包模拟代码。代码使用 split-step Fourier method 求解二维含时薛定谔方程，模拟二维高斯波包穿过方形/矩形方势区域的传播过程，并输出二维综合动画与三维曲面动画。

## Files

* `quantum_tunneling_2d.py`：二维求解器、诊断量计算、二维动画渲染和三维动画渲染代码。
* `1/tunneling_2d.mp4`：二维综合视频，包含概率密度、实部波和虚部波。
* `1/tunneling_2d_3d.mp4`：三维综合视频，包含概率密度、实部波和虚部波的三维曲面展示。
* `1/tunneling_2d_3d_high_sampling.mp4`：更高时间/空间采样的三维预览视频。
* `1/2d_square_well_simulation_summary.docx`：二维方势模拟仿真思路总结文档。

## Dependencies

依赖由项目根目录的 `requirements.txt` 统一管理：

```bash
pip install -r requirements.txt
```

本代码主要依赖：

* `numpy`：二维复波函数、FFT、概率积分等数值计算。
* `matplotlib`：二维热图、三维曲面图和动画导出。
* `ffmpeg`：将动画导出为 `.mp4` 视频。

如果 `ffmpeg` 不在 Python 进程的 PATH 中，可以通过 `--ffmpeg-path` 显式传入 `ffmpeg.exe` 路径。

## Run

在项目根目录运行：

```bash
python contributors/zhb-二维方势/quantum_tunneling_2d.py
```

如果是在本目录内直接运行：

```bash
python quantum_tunneling_2d.py
```

推荐显式指定输出文件：

```bash
python quantum_tunneling_2d.py ^
  --output 1/tunneling_2d.mp4 ^
  --output-3d 1/tunneling_2d_3d.mp4
```

如果需要指定 ffmpeg：

```bash
python quantum_tunneling_2d.py ^
  --ffmpeg-path C:\path\to\ffmpeg.exe ^
  --output 1/tunneling_2d.mp4 ^
  --output-3d 1/tunneling_2d_3d.mp4
```

## Output

默认会先完成一次二维量子波包数值模拟，然后基于同一组模拟数据导出两个视频：

```text
1/tunneling_2d.mp4
1/tunneling_2d_3d.mp4
```

二维视频包含三个面板：

* 概率密度 `|psi|^2`
* 实部波 `|Re(psi)|`
* 虚部波 `|Im(psi)|`

三维视频同样包含三个面板，只是将上述三个量作为三维曲面展示，其中 x-y 平面为空间位置，z 轴表示对应物理量的幅值。

## Model

模拟对象为二维复波函数：

```text
psi(x, y, t)
```

其演化满足二维含时薛定谔方程：

```text
i hbar dpsi/dt =
[-hbar^2/(2m)(d^2/dx^2 + d^2/dy^2) + V(x,y)] psi
```

初始波函数采用二维高斯波包：

```text
psi(x,y,0) = Gaussian(x,y) * exp[i(kx x + ky y)]
```

其中：

* `packet_x0`, `packet_y0` 控制初始波包中心。
* `packet_sigma_x`, `packet_sigma_y` 控制波包宽度。
* `packet_kx0`, `packet_ky0` 控制入射方向和平均动量。

二维方势区域由矩形条件定义：

```text
|x - barrier_x0| <= barrier_x_width / 2
|y - barrier_y0| <= barrier_y_width / 2
```

当前实现中该区域作为二维方势垒使用；若需要方势阱模型，可将势能高度改为负值。

## Numerical Method

代码采用 split-step Fourier method 进行时间演化。哈密顿量分为动能项和势能项：

```text
H = T + V
```

一个时间步内使用 Strang splitting：

```text
exp(-iHdt/hbar)
≈ exp(-iVdt/2hbar) exp(-iTdt/hbar) exp(-iVdt/2hbar)
```

实现流程：

1. 在位置空间乘以半步势能相位。
2. 使用二维 FFT 变换到动量空间。
3. 在动量空间乘以动能相位，动能相位依赖 `kx^2 + ky^2`。
4. 使用二维 IFFT 回到位置空间。
5. 再乘以半步势能相位。
6. 乘以边界吸收 mask，减少 FFT 周期边界带来的回卷伪影。

## Useful Parameters

增加透射：

```bash
python quantum_tunneling_2d.py --barrier-height 0.75 --barrier-x-width 12 --barrier-y-width 24 --kx0 1.45
```

减少透射：

```bash
python quantum_tunneling_2d.py --barrier-height 1.20 --barrier-x-width 24 --barrier-y-width 40 --kx0 1.30
```

减少边缘绕射：

```bash
python quantum_tunneling_2d.py --barrier-y-width 48 --sigma-y 8
```

生成更平滑的三维动画：

```bash
python quantum_tunneling_2d.py --three-d-frame-step 1 --three-d-grid-step 2
```

快速预览：

```bash
python quantum_tunneling_2d.py --frames 80 --steps-per-frame 3 --fps 24 --output 1/preview_2d.gif --output-3d 1/preview_3d.gif
```

## Diagnostics

动画左上角会显示以下诊断量：

* `R`：反射概率。
* `T`：透射概率。
* `bypass`：从势场边缘绕过的概率估计。
* `near`：仍停留在势场附近的概率。
* `norm`：当前总概率范数。

这些量都由概率密度 `|psi|^2` 在不同空间区域上的积分得到。
