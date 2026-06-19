[README.md](https://github.com/user-attachments/files/29135696/README.md)
# zhb - 二维方势

本目录保存 zhb 负责的二维方势量子隧穿模拟代码。代码使用 split-step Fourier method 求解二维含时薛定谔方程，模拟二维高斯波包穿过方形或矩形方势区域时的反射、透射、边缘绕射和波函数相位结构变化。

## Files

* `quantum_tunneling_2d.py`：二维方势模拟主程序，包含数值演化、诊断量计算、二维动画和三维动画导出。
* `README.md`：本说明文件。

## Dependencies

依赖由项目根目录的 `requirements.txt` 统一管理：

```powershell
pip install -r requirements.txt
```

主要依赖：

* `numpy`：二维复波函数、FFT 和概率积分。
* `matplotlib`：二维热图、三维曲面图和动画生成。
* `ffmpeg`：导出 `.mp4` 视频。

如果 `ffmpeg` 不在环境变量 PATH 中，可以使用 `--ffmpeg-path` 手动指定 `ffmpeg.exe`。

## Run

在项目根目录运行：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py
```

如果已经进入本目录，也可以运行：

```powershell
python quantum_tunneling_2d.py
```

默认输出写入：

```text
output/tunneling_2d.mp4
output/tunneling_2d_3d.mp4
```

如果需要手动指定 ffmpeg：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py `
  --ffmpeg-path "C:\path\to\ffmpeg.exe"
```

也可以显式指定输出路径：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py `
  --output output/tunneling_2d.mp4 `
  --output-3d output/tunneling_2d_3d.mp4
```

## Output

程序会先完成一次二维波函数数值模拟，然后基于同一组数据导出两个视频：

* `output/tunneling_2d.mp4`：二维综合动画。
* `output/tunneling_2d_3d.mp4`：三维综合动画。

二维动画包含三个面板：

* 概率密度 `|psi|^2`
* 实部波 `|Re(psi)|`
* 虚部波 `|Im(psi)|`

三维动画也包含三个面板，只是将上述物理量作为三维曲面展示。其中 x-y 平面表示二维空间位置，z 轴表示对应物理量的幅值。

## Model

模拟对象是二维复波函数：

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

当前参数下该区域作为二维方势垒使用；如果需要模拟方势阱，可以将势能高度改为负值。

## Numerical Method

代码采用 split-step Fourier method 进行时间演化。哈密顿量写为：

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
6. 乘以边界吸收 mask，减少 FFT 周期边界导致的回卷伪影。

## Useful Parameters

增加透射：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py --barrier-height 0.75 --barrier-x-width 12 --barrier-y-width 24 --kx0 1.45
```

减少透射：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py --barrier-height 1.20 --barrier-x-width 24 --barrier-y-width 40 --kx0 1.30
```

减少边缘绕射：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py --barrier-y-width 48 --sigma-y 8
```

生成更平滑的三维动画：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py --three-d-frame-step 1 --three-d-grid-step 2
```

## Diagnostics

动画中会显示以下诊断量：

* `R`：反射概率。
* `T`：透射概率。
* `bypass`：从势场边缘绕过的概率估计。
* `near`：仍停留在势场附近的概率。
* `norm`：当前总概率范数。

这些量均由概率密度 `|psi|^2` 在不同空间区域上的积分得到。
