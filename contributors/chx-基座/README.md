# chx - 基座

本目录保存 chx 负责的量子隧穿模拟基座代码。代码使用 split-step Fourier method 模拟一维高斯波包穿过矩形势垒，并渲染三维复波函数动画。

## Files

- `quantum_tunneling_3d.py`：求解器、诊断计算和动画渲染代码。

## Dependencies

依赖由项目根目录的 `requirements.txt` 统一管理：

```powershell
pip install -r requirements.txt
```

## Run

在项目根目录运行：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py
```

默认输出：

```text
output/tunneling_3d.mp4
```

快速预览：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py --frames 60 --steps-per-frame 8 --output output/preview.gif
```

如果 `ffmpeg` 不在 Python 进程的 PATH 中，可以显式传入路径：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py --ffmpeg-path C:\path\to\ffmpeg.exe
```

## Useful Parameters

增加隧穿：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py --barrier-height 1.0 --barrier-width 2.8 --k0 1.4
```

减少隧穿：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py --barrier-height 1.6 --barrier-width 7 --k0 1.25
```

生成更平滑的动画：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py --frames 300 --steps-per-frame 5 --fps 48
```
