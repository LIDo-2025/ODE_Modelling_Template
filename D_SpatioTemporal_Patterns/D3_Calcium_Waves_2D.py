"""
D3 · Calcium Waves in 2D — Target Patterns and Spiral Waves
============================================================
Animated simulation of the simplified CICR model on a 2-D grid.

HOW TO USE
----------
Set MODE at the top of this file, then run the script:

    MODE = "target"   → concentric target rings from a central pacemaker
    MODE = "spiral"   → rotating spiral wave (phase-shifted pacemaker)

The animation window updates live.  Close it to stop.

MODEL
-----
Same calcium–IP3 kinetics as D1 and D2:

    dX/dt = a - m2·X/(1+X) + [m3·Y/(k1+Y)]·X²/(ka+X²) + Y − k·X
    dY/dt =     m2·X/(1+X) − [m3·Y/(k1+Y)]·X²/(ka+X²) − Y

with 2-D diffusion of Ca²⁺ (X) added via the discrete Laplacian.

The central disc receives a periodic Ca²⁺ stimulus.
  • Symmetric stimulus (phase_shift=0) → target pattern.
  • Angle-dependent phase  (phase_shift>0) → spiral wave.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ── ① Choose what to run ──────────────────────────────────────────────────────
MODE = "target"     # "target"  or  "spiral"

# ── ② Simulation settings ─────────────────────────────────────────────────────
L          = 128    # grid size (L × L cells)
N_STEPS    = 8000   # total integration steps
PLOT_EVERY = 5      # redraw every N steps  (lower = smoother, slower)
SEED       = 42     # random seed for reproducible initial noise

# ── ③ Pacemaker settings ──────────────────────────────────────────────────────
STIM_AMP   = 1.0    # stimulus amplitude
STIM_FREQ  = 0.11   # stimulus frequency  (pacemaker period ≈ 1/STIM_FREQ)
STIM_R     = 4      # pacemaker disc radius (cells)

# phase_shift = 0   → target;   phase_shift = 0.3 → clean single-arm spiral
PHASE_SHIFT = 0.0 if MODE == "target" else 0.3

# ── ④ Model parameters ────────────────────────────────────────────────────────
A   = 0.33    # Ca²⁺ inflow (oscillatory bulk)
M2  = 20.0
M3  = 23.0
KA  = 0.8
K   = 0.8
K1  = 0.8
D_X = 0.5     # Ca²⁺ diffusion coefficient
DT  = 0.05    # time step
DX  = 1.0     # grid spacing


# ── Model class ───────────────────────────────────────────────────────────────

class CalciumGrid:
    """2-D CICR model with a central periodic pacemaker disc."""

    def __init__(self):
        self.X = np.full((L, L), 0.4)    # Ca²⁺ field
        self.Y = np.full((L, L), 2.7)    # IP₃ field
        self.t = 0.0

        # Reproducible noise on initial conditions
        rng = np.random.default_rng(SEED)
        self.X += rng.normal(0, 0.05, self.X.shape)
        self.Y += rng.normal(0, 0.02, self.Y.shape)
        self.X = np.maximum(self.X, 0.0)
        self.Y = np.maximum(self.Y, 0.0)

        # Laplacian kernel (no-flux via reflect padding in ndimage)
        self._lap = np.array([[0, 1, 0],
                              [1,-4, 1],
                              [0, 1, 0]], dtype=float) / DX**2

        # Pre-compute pacemaker mask and phase map
        cx = cy = L // 2
        jj, ii = np.meshgrid(np.arange(L), np.arange(L))
        dist = np.sqrt((ii - cx)**2 + (jj - cy)**2)
        self._mask = dist <= STIM_R

        if PHASE_SHIFT != 0.0:
            angle = np.arctan2(jj - cy, ii - cx)
            self._phase_map = np.where(self._mask,
                                       angle * PHASE_SHIFT, 0.0)
        else:
            self._phase_map = None

    def step(self):
        X, Y = self.X, self.Y

        # Reaction
        ip3 = (M3 * Y / (K1 + Y)) * X**2 / (KA + X**2)
        dX  = A - M2*X/(1+X) + ip3 + Y - K*X
        dY  =     M2*X/(1+X) - ip3     - Y

        # Diffusion (reflect = no-flux boundaries)
        dX += D_X * ndimage.convolve(X, self._lap, mode='reflect')

        # Forward Euler
        self.X = np.maximum(X + DT * dX, 0.0)
        self.Y = np.maximum(Y + DT * dY, 0.0)

        # Pacemaker stimulus
        if self._phase_map is not None:
            stim = STIM_AMP * np.sin(
                2*np.pi * STIM_FREQ * self.t + self._phase_map)
            self.X[self._mask] += stim[self._mask] * DT
        else:
            stim_val = STIM_AMP * np.sin(2*np.pi * STIM_FREQ * self.t)
            self.X[self._mask] += stim_val * DT

        self.X = np.maximum(self.X, 0.0)
        self.t += DT


# ── Animation ─────────────────────────────────────────────────────────────────

def run():
    grid = CalciumGrid()
    pattern = "TARGET PATTERN" if MODE == "target" else "SPIRAL WAVE"
    print(f"Running {pattern}  ({L}×{L} grid, {N_STEPS} steps)")
    print(f"phase_shift = {PHASE_SHIFT:.2f}π  |  D_X = {D_X}  |  seed = {SEED}")

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')

    im = ax.imshow(grid.X, cmap='hot', origin='lower',
                   vmin=0.0, vmax=1.2, interpolation='bilinear')
    title = ax.set_title(f'{pattern}   t = 0.0',
                         color='white', fontsize=13, pad=8)
    plt.colorbar(im, ax=ax, label='[Ca²⁺]',
                 fraction=0.046, pad=0.02).ax.yaxis.label.set_color('white')
    plt.tight_layout()

    for step in range(N_STEPS):
        grid.step()

        if step % PLOT_EVERY == 0:
            im.set_data(grid.X)
            title.set_text(f'{pattern}   t = {grid.t:.1f}')
            plt.pause(0.001)

        if step % 1000 == 0:
            print(f"  step {step:5d}/{N_STEPS}  t={grid.t:6.1f}"
                  f"  X∈[{grid.X.min():.2f}, {grid.X.max():.2f}]")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    run()
