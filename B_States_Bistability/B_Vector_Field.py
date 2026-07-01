"""
B · Animated Vector Field — Toggle Switch Bistability
=====================================================
Streamlines grow from random seeds, coloured by flow speed at each tip
(dark = slow, bright = fast).  Nullclines, fixed points with attractor glow,
and the separatrix are drawn as a static background.

Teaching intent
---------------
* The phase plane is a *fluid medium*, not an empty backdrop.
* Fixed points are *dynamically maintained* attractors — the flow arrives
  continuously and is held there by the competing forces in the open system.
* The separatrix (gold dashed) divides the plane into two basins of attraction;
  it is the "tipping boundary" between states.

Controls
--------
  α slider   — maximal transcription rate  (lower → bistability lost)
  n slider   — Hill coefficient / cooperativity  (n > 1 required)
  [Pause]    — freeze animation and show arrowheads on every streamline tip
  [Regenerate] — rebuild after moving a slider
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.optimize import fsolve

EXPORT_HTML = False   # set True to save animation as HTML instead of showing interactively

# ── Colour theme ─────────────────────────────────────────────────────────────
BG_FIG   = '#12121f'   # figure background
BG_AXES  = '#080814'   # axes background
CLR_TXT  = '#d8d8f0'   # all text / tick labels
CLR_NC_U = '#38bdf8'   # u-nullcline  (sky blue)
CLR_NC_V = '#f87171'   # v-nullcline  (rose)
CLR_SEP  = '#fbbf24'   # separatrix   (amber)
CMAP     = plt.cm.plasma

# ── Model parameters ─────────────────────────────────────────────────────────
ALPHA = 10.0
N     = 3

GRID_MIN, GRID_MAX = -0.5, 12.0
GRID_PTS           = 62

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def toggle_rhs(u, v, alpha, n):
    uc = max(0.0, u)
    vc = max(0.0, v)
    return (alpha / (1 + vc**n) - u,
            alpha / (1 + uc**n) - v)


def _jacobian(fp, alpha, n):
    """2×2 Jacobian matrix at fp."""
    u, v = max(fp[0], 1e-9), max(fp[1], 1e-9)
    dfu_dv = -alpha * n * v**(n-1) / (1 + v**n)**2
    dfv_du = -alpha * n * u**(n-1) / (1 + u**n)**2
    return np.array([[-1.0, dfu_dv],
                     [dfv_du, -1.0]])


def jacobian_det(fp, alpha, n):
    J = _jacobian(fp, alpha, n)
    return np.linalg.det(J)


def classify_fp(fp, alpha, n):
    det = jacobian_det(fp, alpha, n)
    if det < 0:
        return 's', 'white',    'Saddle'
    elif fp[0] >= fp[1]:
        return 'o', CLR_NC_U,  'Attractor  (u high)'
    else:
        return 'o', CLR_NC_V,  'Attractor  (v high)'


def find_fixed_points(alpha, n, within_plot=False):
    def rhs(y): return toggle_rhs(y[0], y[1], alpha, n)
    fps = []
    for guess in [[alpha*0.95, 0.1], [0.1, alpha*0.95],
                  [alpha**0.5, alpha**0.5], [alpha*0.5, alpha*0.5]]:
        fp, _, ier, _ = fsolve(rhs, guess, full_output=True)
        if ier == 1 and fp[0] >= GRID_MIN and fp[1] >= GRID_MIN:
            if all(np.linalg.norm(fp - f) > 0.05 for f in fps):
                fps.append(fp)
    if within_plot:
        fps = [fp for fp in fps if fp[0] <= GRID_MAX and fp[1] <= GRID_MAX]
    return fps


# ─────────────────────────────────────────────────────────────────────────────
# Separatrix  (stable manifold of the saddle)
# ─────────────────────────────────────────────────────────────────────────────

def compute_separatrix(alpha, n, eps=0.08, dt=0.04, max_steps=600):
    """
    Backward-integrate from saddle ± eps along the stable eigenvector.
    Returns (saddle_point, [branch_array, ...]) or (None, []).
    """
    fps = find_fixed_points(alpha, n, within_plot=False)
    saddle = next((fp for fp in fps if jacobian_det(fp, alpha, n) < 0), None)
    if saddle is None:
        return None, []

    J = _jacobian(saddle, alpha, n)
    eigenvalues, eigenvectors = np.linalg.eig(J)
    # Stable manifold: eigenvector whose eigenvalue has the most negative real part
    stable_idx = np.argmin(eigenvalues.real)
    sv = eigenvectors[:, stable_idx].real
    sv /= np.linalg.norm(sv)

    branches = []
    for sign in [+1, -1]:
        pt = saddle + sign * eps * sv
        pts = [pt.copy()]
        for _ in range(max_steps):
            dx, dy = toggle_rhs(pt[0], pt[1], alpha, n)
            pt = pt - np.array([dx, dy]) * dt   # backward integration
            if not (GRID_MIN <= pt[0] <= GRID_MAX and
                    GRID_MIN <= pt[1] <= GRID_MAX):
                break
            pts.append(pt.copy())
        if len(pts) > 2:
            branches.append(np.array(pts))

    return saddle, branches


# ─────────────────────────────────────────────────────────────────────────────
# Vector field
# ─────────────────────────────────────────────────────────────────────────────

def compute_field(alpha, n):
    x = np.linspace(GRID_MIN, GRID_MAX, GRID_PTS)
    y = np.linspace(GRID_MIN, GRID_MAX, GRID_PTS)
    X, Y = np.meshgrid(x, y)
    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            DX[i,j], DY[i,j] = toggle_rhs(X[i,j], Y[i,j], alpha, n)
    magnitudes = np.sqrt(DX**2 + DY**2)
    speed = magnitudes.copy(); speed[speed == 0] = 1
    return x, y, magnitudes, DX/speed, DY/speed


# ─────────────────────────────────────────────────────────────────────────────
# Streamlines
# ─────────────────────────────────────────────────────────────────────────────

def generate_streamlines(x, y, DX_norm, DY_norm, magnitudes,
                          num_streams=500, max_length=4.5, seed=42):
    rng = np.random.default_rng(seed)
    streamlines = []
    x_starts = rng.uniform(GRID_MIN, GRID_MAX, num_streams)
    y_starts = rng.uniform(GRID_MIN, GRID_MAX, num_streams)
    dt = 0.15

    for x0, y0 in zip(x_starts, y_starts):
        pts = [[x0, y0]]; cols = []
        xc, yc = x0, y0
        for _ in range(int(max_length / dt)):
            if not (GRID_MIN <= xc <= GRID_MAX and GRID_MIN <= yc <= GRID_MAX):
                break
            i = np.searchsorted(x, xc) - 1
            j = np.searchsorted(y, yc) - 1
            if not (0 <= i < len(x)-1 and 0 <= j < len(y)-1):
                break
            wx = (xc - x[i]) / (x[i+1] - x[i])
            wy = (yc - y[j]) / (y[j+1] - y[j])
            def blerp(F, wx=wx, wy=wy, i=i, j=j):
                return ((1-wx)*(1-wy)*F[j,i]   + wx*(1-wy)*F[j,i+1] +
                        (1-wx)*   wy *F[j+1,i] + wx*   wy *F[j+1,i+1])
            dx = blerp(DX_norm); dy = blerp(DY_norm)
            xc += dx*dt; yc += dy*dt
            pts.append([xc, yc])
            cols.append(blerp(magnitudes))
        if len(pts) > 2:
            streamlines.append({'points': np.array(pts),
                                 'colors': np.array(cols)})
    return streamlines


# ─────────────────────────────────────────────────────────────────────────────
# Initial computation
# ─────────────────────────────────────────────────────────────────────────────

print("Computing vector field …")
x_grid, y_grid, magnitudes, DX_norm, DY_norm = compute_field(ALPHA, N)
print("Generating streamlines …")
streamlines = generate_streamlines(x_grid, y_grid, DX_norm, DY_norm, magnitudes)
print("Computing separatrix …")
saddle_pt, sep_branches = compute_separatrix(ALPHA, N)
print(f"{len(streamlines)} streamlines ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'text.color':      CLR_TXT,
    'axes.labelcolor': CLR_TXT,
    'xtick.color':     CLR_TXT,
    'ytick.color':     CLR_TXT,
})

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AXES)
plt.subplots_adjust(bottom=0.08 if EXPORT_HTML else 0.28, left=0.10, right=0.88)

ax.set_xlabel('Protein  u', fontsize=13)
ax.set_ylabel('Protein  v', fontsize=13)
ax.set_xlim(GRID_MIN, GRID_MAX)
ax.set_ylim(GRID_MIN, GRID_MAX)
for spine in ax.spines.values():
    spine.set_edgecolor('#333355')
ax.grid(True, alpha=0.07, color='white', linestyle='--')

# Colorbar
vmin_global = magnitudes.min()
vmax_global = magnitudes.max()
sm = plt.cm.ScalarMappable(cmap=CMAP,
     norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Flow speed', shrink=0.85)
cbar.ax.yaxis.label.set_color(CLR_TXT)
cbar.ax.tick_params(colors=CLR_TXT)
cbar.outline.set_edgecolor('#333355')


# ─────────────────────────────────────────────────────────────────────────────
# Static background: nullclines, separatrix, fixed points
# ─────────────────────────────────────────────────────────────────────────────

bg_artists = []

def draw_background(alpha, n):
    global bg_artists, saddle_pt, sep_branches

    for a in bg_artists:
        try: a.remove()
        except: pass
    bg_artists = []

    # Nullclines
    v_r = np.linspace(GRID_MIN, GRID_MAX, 400)
    u_r = np.linspace(GRID_MIN, GRID_MAX, 400)
    l1, = ax.plot(alpha / (1 + v_r**n), v_r,
                  color=CLR_NC_U, linewidth=2.0, alpha=1.0, zorder=3,
                  label='u-nullcline  (du/dt = 0)')
    l2, = ax.plot(u_r, alpha / (1 + u_r**n),
                  color=CLR_NC_V, linewidth=2.0, alpha=1.0, zorder=3,
                  label='v-nullcline  (dv/dt = 0)')
    bg_artists += [l1, l2]

    # Separatrix
    saddle_pt, sep_branches = compute_separatrix(alpha, n)
    sep_added_to_legend = False
    for branch in sep_branches:
        lbl = 'Separatrix' if not sep_added_to_legend else '_nolegend_'
        sep_added_to_legend = True
        sl, = ax.plot(branch[:, 0], branch[:, 1],
                      color=CLR_SEP, linewidth=1.8, linestyle='--',
                      alpha=0.75, zorder=4, label=lbl)
        bg_artists.append(sl)

    # Fixed points — glow ring + solid marker
    fps_visible = find_fixed_points(alpha, n, within_plot=True)
    seen_labels = set()
    for fp in fps_visible:
        marker, color, label = classify_fp(fp, alpha, n)
        # outer glow
        glow = ax.scatter(*fp, s=700, color=color, alpha=0.10,
                          marker='o', zorder=4, edgecolors='none')
        # inner glow
        glow2 = ax.scatter(*fp, s=300, color=color, alpha=0.18,
                           marker='o', zorder=5, edgecolors='none')
        # main marker
        lbl = label if label not in seen_labels else '_nolegend_'
        seen_labels.add(label)
        sc = ax.scatter(*fp, s=130, color=color,
                        edgecolors='white', linewidths=1.2,
                        marker=marker, zorder=6, label=lbl)
        bg_artists += [glow, glow2, sc]

    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.45,
                    facecolor='#1a1a2e', edgecolor='#444466')
    bg_artists.append(leg)

    # Title
    fps_all  = find_fixed_points(alpha, n, within_plot=False)
    n_stable = sum(1 for fp in fps_all if jacobian_det(fp, alpha, n) > 0)
    bistable = n_stable >= 2
    ax.set_title(
        f'Toggle Switch  |  α = {alpha:.1f},  n = {int(n)}  '
        f'→  {"BISTABLE" if bistable else "monostable"}',
        fontsize=13,
        color='#6ee7b7' if bistable else '#fca5a5',
    )

draw_background(ALPHA, N)


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

NUM_FRAMES       = 60
animation_running = True
current_frame    = [0]

lines_art = []
for stream in streamlines:
    line, = ax.plot([], [], linewidth=1.8, alpha=0.70, zorder=2)
    lines_art.append(line)

pause_quiver = [None]


def update_frame():
    progress = current_frame[0] / NUM_FRAMES
    for line, stream in zip(lines_art, streamlines):
        pts  = stream['points']
        cols = stream['colors']
        n_pts = max(2, int(len(pts) * progress))
        line.set_data(pts[:n_pts, 0], pts[:n_pts, 1])
        if len(cols) > 0:
            # Colour by *tip* speed: dark where flow stalls near attractors
            tip_c  = cols[n_pts - 2] if n_pts > 1 else cols[0]
            norm_c = (tip_c - vmin_global) / max(vmax_global - vmin_global, 1e-9)
            line.set_color(CMAP(norm_c))
    if animation_running:
        current_frame[0] = (current_frame[0] + 1) % NUM_FRAMES
    fig.canvas.draw_idle()


update_frame()
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(update_frame)
timer.start()


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild after parameter change
# ─────────────────────────────────────────────────────────────────────────────

def rebuild(alpha, n):
    global streamlines, lines_art, magnitudes, vmin_global, vmax_global

    if pause_quiver[0] is not None:
        pause_quiver[0].remove()
        pause_quiver[0] = None

    print(f"Rebuilding for α = {alpha:.1f},  n = {int(n)} …")
    x_g, y_g, mags, DXn, DYn = compute_field(alpha, n)
    streams_new = generate_streamlines(x_g, y_g, DXn, DYn, mags)

    for l in lines_art:
        l.remove()
    lines_art.clear(); streamlines.clear()
    streamlines.extend(streams_new)
    magnitudes = mags
    vmin_global = mags.min(); vmax_global = mags.max()

    for stream in streamlines:
        line, = ax.plot([], [], linewidth=1.8, alpha=0.70, zorder=2)
        lines_art.append(line)

    draw_background(alpha, n)
    current_frame[0] = 0
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Widgets
# ─────────────────────────────────────────────────────────────────────────────

# ── Launch ───────────────────────────────────────────────────────────────────
if EXPORT_HTML:
    from matplotlib.animation import FuncAnimation

    def _export_update(fi):
        current_frame[0] = fi          # FuncAnimation drives the frame counter
        update_frame()                 # animation_running=True so lines grow normally

    anim = FuncAnimation(fig, _export_update, frames=NUM_FRAMES,
                         interval=50, blit=False)
    out  = Path(__file__).parent / 'Export' / 'B_Vector_Field.html'
    out.parent.mkdir(exist_ok=True)
    print(f"Rendering {NUM_FRAMES} frames to HTML …")
    out.write_text(anim.to_jshtml(fps=20, default_mode='loop'))
    print(f"Saved → {out}")
    plt.close()

else:
    ax_btn_pause = plt.axes([0.10, 0.03, 0.12, 0.04])
    ax_btn_regen = plt.axes([0.78, 0.03, 0.12, 0.04])
    ax_sl_alpha  = plt.axes([0.25, 0.14, 0.50, 0.025])
    ax_sl_n      = plt.axes([0.25, 0.09, 0.50, 0.025])

    btn_pause = Button(ax_btn_pause, 'Pause')
    btn_regen = Button(ax_btn_regen, 'Regenerate',
                       color='#1e3a2a', hovercolor='#2d5a3a')
    sl_alpha  = Slider(ax_sl_alpha, 'α  (max. expression)', 1.0, 20.0,
                       valinit=ALPHA, valstep=0.5)
    sl_n      = Slider(ax_sl_n,     'n  (cooperativity)',   1,   5,
                       valinit=N,    valstep=1)

    def toggle_pause(event):
        global animation_running
        animation_running = not animation_running
        btn_pause.label.set_text('Play' if not animation_running else 'Pause')

        if not animation_running:
            progress = current_frame[0] / NUM_FRAMES
            X_t, Y_t, U_a, V_a, C_a = [], [], [], [], []
            for stream in streamlines:
                pts  = stream['points']
                cols = stream['colors']
                n_pts = max(2, int(len(pts) * progress))
                if n_pts >= 2:
                    tip  = pts[n_pts - 1]
                    prev = pts[n_pts - 2]
                    d = tip - prev
                    L = np.linalg.norm(d)
                    if L > 1e-9:
                        X_t.append(tip[0]);  Y_t.append(tip[1])
                        U_a.append(d[0]/L);  V_a.append(d[1]/L)
                        tip_c  = cols[n_pts-2] if len(cols) > 0 else vmin_global
                        norm_c = (tip_c - vmin_global) / max(vmax_global - vmin_global, 1e-9)
                        C_a.append(CMAP(norm_c))
            if X_t:
                pause_quiver[0] = ax.quiver(
                    X_t, Y_t, U_a, V_a,
                    color=C_a, alpha=0.85,
                    scale=22, width=0.004,
                    headwidth=4, headlength=5, zorder=5,
                )
        else:
            if pause_quiver[0] is not None:
                pause_quiver[0].remove()
                pause_quiver[0] = None

        update_frame()

    def on_regenerate(event):
        rebuild(sl_alpha.val, int(sl_n.val))

    btn_pause.on_clicked(toggle_pause)
    btn_regen.on_clicked(on_regenerate)

    update_frame()
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(update_frame)
    timer.start()
    plt.show()
