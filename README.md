# 3D Quantum Tunneling Animation

This project simulates a Gaussian wave packet scattering from a 1D rectangular
potential barrier and renders the complex wavefunction as a fixed-view 3D curve:

- `x`: position along the 1D system
- `Re(psi)`: real axis of the wavefunction
- `Im(psi)`: imaginary axis of the wavefunction

The animation also draws two projection curves: the real part on the bottom
`x-Re(psi)` plane and the imaginary part on the front `x-Im(psi)` plane.

The solver uses the split-step Fourier method for the time-dependent
Schrodinger equation in natural units:

```text
i dpsi/dt = [-1/2 d^2/dx^2 + V(x)] psi
```

The default packet energy is below the barrier height, so the animation shows
partial reflection and a nonzero transmitted tail.

## Run

The current machine already has the required libraries installed:

```powershell
python quantum_tunneling_3d.py
```

The default output is:

```text
output/tunneling_3d.mp4
```

For a faster preview:

```powershell
python quantum_tunneling_3d.py --frames 60 --steps-per-frame 8 --output output/preview.gif
```

If `ffmpeg` is installed, write MP4 instead:

```powershell
python quantum_tunneling_3d.py --output output/tunneling_3d.mp4
```

If `ffmpeg` is not on the Python process PATH, pass its path explicitly:

```powershell
python quantum_tunneling_3d.py --ffmpeg-path C:\path\to\ffmpeg.exe
```

For full chroma 4:4:4 MP4 output:

```powershell
python quantum_tunneling_3d.py --pix-fmt yuv444p
```

## Useful Parameters

Increase tunneling:

```powershell
python quantum_tunneling_3d.py --barrier-height 1.0 --barrier-width 2.8 --k0 1.4
```

Decrease tunneling:

```powershell
python quantum_tunneling_3d.py --barrier-height 1.6 --barrier-width 7 --k0 1.25
```

Make a smoother animation:

```powershell
python quantum_tunneling_3d.py --frames 300 --steps-per-frame 5 --fps 48
```

## Method Notes

At each small time step, the code applies a symmetric Strang split:

```text
psi(t + dt) ~= exp(-i V dt/2) FFT^-1[
                 exp(-i k^2 dt / 2m) FFT[
                   exp(-i V dt/2) psi(t)
                 ]
               ]
```

The potential is diagonal in position space, while the kinetic operator is
diagonal in momentum space. A weak absorbing boundary is applied near the ends
of the domain to reduce FFT periodic wrap-around artifacts.

## Files

- `quantum_tunneling_3d.py`: solver, diagnostics, and animation renderer
- `Modeling Quantum Tunneling With Python _ by Rhett Allain _ Medium.html`:
  local reference article supplied in the workspace
