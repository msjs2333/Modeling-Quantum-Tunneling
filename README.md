# Modeling Quantum Tunneling

团队量子隧穿建模项目仓库。根目录保留项目级说明、依赖文件和协作目录；各成员的代码与说明按贡献者归档。

## File Structure

```text
.
├── README.md
├── requirements.txt
├── contributors/
│   ├── chx-基座/
│   │   ├── README.md
│   │   └── quantum_tunneling_3d.py
│   └── zhb-二维方势/
│       ├── README.md
│       └── quantum_tunneling_2d.py
└── ppt/
    └── README.md
```

## Contributors

| 成员 | 内容 | 目录 | 说明 |
| --- | --- | --- | --- |
| chx | 基座 | `contributors/chx-基座/` | 量子隧穿模拟基座代码 |
| zhb | 二维方势 | `contributors/zhb-二维方势/` | 量子隧穿模拟二维方势代码 |
| 待填写 | 待填写 | `contributors/待填写-待填写/` | 待填写 |

## Directory Rules

- `contributors/`：存放团队成员贡献内容，每个子目录使用“成员名-内容”命名。
- `contributors/chx-基座/`：chx 的基座代码与对应自述。
- `contributors/zhb-二维方势`: zhb 的二维方势代码与对应自述。
- `ppt/`：后续放置展示、汇报或答辩用 PPT 文件。
- `requirements.txt`：项目级 Python 依赖。

## Usage

安装依赖：

```powershell
pip install -r requirements.txt
```

运行 chx 的基座代码：

```powershell
python contributors/chx-基座/quantum_tunneling_3d.py
```

默认输出写入：

```text
output/tunneling_3d.mp4
```

运行 zhb 的二维方势代码：

```powershell
python contributors/zhb-二维方势/quantum_tunneling_2d.py
```

默认输出写入：

```text
contributors/zhb-二维方势/1/tunneling_2d.mp4
contributors/zhb-二维方势/1/tunneling_2d_3d.mp4
```
