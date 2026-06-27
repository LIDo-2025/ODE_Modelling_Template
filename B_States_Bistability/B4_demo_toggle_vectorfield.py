"""
Animated vector-field demo — Toggle Switch bistability
======================================================
Streamlines grow from random seeds, coloured by flow speed (dark = slow, bright = fast).
The nullclines and fixed points are drawn as a static background.

Sliders:
  alpha  — maximal transcription rate     (controls whether bistability exists)
  n      — Hill coefficient (cooperativity)  (must be > 1 for bistability)

Use the [Regenerate] button after moving a slider to rebuild the streamlines.
Play / Pause freezes the animation and shows arrowheads at each tip.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.optimize import fsolve

# ── Model ────────────────────────────────────────────────────────────────────

ALPHA = 10.0
N     = 3

GRID_MIN, GRID_MAX = -0.5, 12.0
GRID_PTS           = 62          # kept ~same density as before

def toggle_rhs(u, v, alpha, n):
    # Clamp to zero: concentrations can't be negative.
    # Near zero the flow points back into the positive quadrant — biologically correct.
    uc = max(0.0, u)
    vc = max(0.0, v)
    return (alpha / (1 + vc**n) - u,
            alpha / (1 + uc**n) - v)

def jacobian_det(fp, alpha, n):
    """det(J) at a fixed point.  Negative → saddle; positive → stable node."""
    u, v = max(fp[0], 1e-9), max(fp[1], 1e-9)
    dfu_dv = -alpha * n * v**(n-1) / (1 + v**n)**2   # ∂(du/dt)/∂v
    dfv_du = -alpha * n * u**(n-1) / (1 + u**n)**2   # ∂(dv/dt)/∂u
    # Jacobian: [[-1, dfu_dv], [dfv_du, -1]]  → det = 1 - dfu_dv*dfv_du
    return 1.0 - dfu_dv * dfv_du

def classify_fp(fp, alpha, n):
    """Return (marker, color, label) based on Jacobian stability."""
    det = jacobian_det(fp, alpha, n)
    if det < 0:
        return 's', 'white',      'Saddle (unstable)'
    elif fp[0] >= fp[1]:
        return 'o', 'deepskyblue','Stable  (u high)'
    else:
        return 'o', 'tomato',     'Stable  (v high)'

def find_fixed_points(alpha, n, within_plot=False):
    """Return fixed points.
    within_plot=False: search unbounded (for bistability classification).
    within_plot=True : only return those visible on the current axes.
    """
    def rhs(y): return toggle_rhs(y[0], y[1], alpha, n)
    fps = []
    for guess in [[alpha*0.95, 0.1], [0.1, alpha*0.95], [alpha**0.5, alpha**0.5],
                  [alpha*0.5, alpha*0.5]]:
        fp, info, ier, _ = fsolve(rhs, guess, full_output=True)
        if ier == 1 and fp[0] >= GRID_MIN and fp[1] >= GRID_MIN:
            if all(np.linalg.norm(fp - f) > 0.05 for f in fps):
                fps.append(fp)
    if within_plot:
        fps = [fp for fp in fps if fp[0] <= GRID_MAX and fp[1] <= GRID_MAX]
    return fps


# ── Vector-field computation ─────────────────────────────────────────────────

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
    DX_norm = DX / speed
    DY_norm = DY / speed
    return x, y, magnitudes, DX_norm, DY_norm


# ── Streamline generation ────────────────────────────────────────────────────

def generate_streamlines(x, y, DX_norm, DY_norm, magnitudes,
                          num_streams=400, max_length=3.5, seed=42):
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
            def blerp(F):
                return ((1-wx)*(1-wy)*F[j,i]   + wx*(1-wy)*F[j,i+1] +
                        (1-wx)*   wy *F[j+1,i] + wx*   wy *F[j+1,i+1])
            dx = blerp(DX_norm); dy = blerp(DY_norm)
            mag = blerp(magnitudes)
            xc += dx*dt; yc += dy*dt
            pts.append([xc, yc]); cols.append(mag)
        if len(pts) > 2:
            streamlines.append({'points': np.array(pts),
                                 'colors': np.array(cols)})
    return streamlines


# ── Initial build ─────────────────────────────────────────────────────────────

print("Computing vector field …")
x_grid, y_grid, magnitudes, DX_norm, DY_norm = compute_field(ALPHA, N)
print("Generating streamlines …")
streamlines = generate_streamlines(x_grid, y_grid, DX_norm, DY_norm, magnitudes)
print(f"{len(streamlines)} streamlines ready.")


# ── Figure layout ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 8))
plt.subplots_adjust(bottom=0.28, left=0.10, right=0.88)

ax.set_xlabel('Protein  u', fontsize=13)
ax.set_ylabel('Protein  v', fontsize=13)
ax.set_xlim(GRID_MIN, GRID_MAX)
ax.set_ylim(GRID_MIN, GRID_MAX)
ax.grid(True, alpha=0.25)

# Colorbar
cmap = plt.cm.plasma
vmin_global = magnitudes.min()
vmax_global = magnitudes.max()
sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Flow speed', shrink=0.85)


# ── Static background: nullclines + fixed points ──────────────────────────────

# We keep handles so we can remove and redraw them after parameter changes
bg_artists = []

def draw_background(alpha, n):
    global bg_artists
    for a in bg_artists:
        try: a.remove()
        except: pass
    bg_artists = []

    v_r = np.linspace(GRID_MIN, GRID_MAX, 400)
    u_r = np.linspace(GRID_MIN, GRID_MAX, 400)
    u_nc = alpha / (1 + v_r**n)   # u-nullcline
    v_nc = alpha / (1 + u_r**n)   # v-nullcline

    l1, = ax.plot(u_nc, v_r, color='deepskyblue', linewidth=2,
                  label='u-nullcline  (du/dt=0)', zorder=3, alpha=0.9)
    l2, = ax.plot(u_r, v_nc, color='tomato',      linewidth=2,
                  label='v-nullcline  (dv/dt=0)', zorder=3, alpha=0.9)
    bg_artists += [l1, l2]

    # Plot only FPs visible within the axes
    fps_visible = find_fixed_points(alpha, n, within_plot=True)
    seen_labels = set()
    for fp in fps_visible:
        marker, color, label = classify_fp(fp, alpha, n)
        lbl = label if label not in seen_labels else '_nolegend_'
        seen_labels.add(label)
        sc = ax.scatter(*fp, s=120, color=color,
                        edgecolors='black', linewidths=1.5,
                        marker=marker, zorder=6, label=lbl)
        bg_artists.append(sc)

    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    bg_artists.append(leg)

    # Bistability check: search full domain (stable FPs may be off-screen)
    fps_all  = find_fixed_points(alpha, n, within_plot=False)
    n_stable = sum(1 for fp in fps_all if jacobian_det(fp, alpha, n) > 0)
    bistable = n_stable >= 2
    title = (f'Toggle Switch  |  α = {alpha:.1f},  n = {int(n)}  '
             f'→  {"BISTABLE" if bistable else "monostable"}')
    ax.set_title(title, fontsize=13,
                 color='darkgreen' if bistable else 'firebrick')

draw_background(ALPHA, N)


# ── Animated streamlines ──────────────────────────────────────────────────────

NUM_FRAMES = 60
animation_running = True
current_frame = [0]

lines = []
for stream in streamlines:
    line, = ax.plot([], [], linewidth=1.5, alpha=0.55, zorder=2)
    lines.append(line)

# Quiver drawn on pause, removed on play
pause_quiver = [None]


def update_frame():
    progress = current_frame[0] / NUM_FRAMES
    for line, stream in zip(lines, streamlines):
        pts  = stream['points']
        cols = stream['colors']
        n_pts = max(2, int(len(pts) * progress))
        line.set_data(pts[:n_pts, 0], pts[:n_pts, 1])
        if len(cols) > 0:
            avg_c = np.mean(cols[:n_pts-1]) if n_pts > 1 else cols[0]
            norm_c = (avg_c - vmin_global) / max(vmax_global - vmin_global, 1e-9)
            line.set_color(cmap(norm_c))
    if animation_running:
        current_frame[0] = (current_frame[0] + 1) % NUM_FRAMES
    fig.canvas.draw_idle()

update_frame()
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(update_frame)
timer.start()


# ── Rebuild after parameter change ───────────────────────────────────────────

def rebuild(alpha, n):
    global streamlines, lines, magnitudes, vmin_global, vmax_global

    # Remove any quiver overlay first
    if pause_quiver[0] is not None:
        pause_quiver[0].remove()
        pause_quiver[0] = None

    print(f"Rebuilding for α={alpha:.1f}, n={int(n)} …")
    x_g, y_g, mags, DXn, DYn = compute_field(alpha, n)
    streams_new = generate_streamlines(x_g, y_g, DXn, DYn, mags)

    for l in lines:
        l.remove()
    lines.clear(); streamlines.clear()
    streamlines.extend(streams_new)
    magnitudes = mags
    vmin_global = mags.min(); vmax_global = mags.max()

    for stream in streamlines:
        line, = ax.plot([], [], linewidth=1.5, alpha=0.55, zorder=2)
        lines.append(line)

    draw_background(alpha, n)
    current_frame[0] = 0
    print("Done.")


# ── Widgets ───────────────────────────────────────────────────────────────────

ax_btn_pause  = plt.axes([0.12, 0.03, 0.12, 0.04])
ax_btn_regen  = plt.axes([0.76, 0.03, 0.12, 0.04])
ax_sl_alpha   = plt.axes([0.12, 0.14, 0.76, 0.025])
ax_sl_n       = plt.axes([0.12, 0.09, 0.76, 0.025])

btn_pause = Button(ax_btn_pause, 'Pause')
btn_regen = Button(ax_btn_regen, 'Regenerate', color='#d4edda', hovercolor='#a8d8a8')
sl_alpha  = Slider(ax_sl_alpha, 'α  (max. expression)', 1.0, 20.0,
                   valinit=ALPHA, valstep=0.5)
sl_n      = Slider(ax_sl_n,     'n  (cooperativity)',   1,   5,
                   valinit=N,    valstep=1)

def toggle_pause(event):
    global animation_running
    animation_running = not animation_running
    btn_pause.label.set_text('Play' if not animation_running else 'Pause')

    if not animation_running:
        # Draw quiver arrowheads at each streamline's current tip
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
                    U_a.append(d[0] / L); V_a.append(d[1] / L)
                    if len(cols) > 0:
                        avg_c = np.mean(cols[:n_pts - 1])
                        norm_c = (avg_c - vmin_global) / max(vmax_global - vmin_global, 1e-9)
                        C_a.append(cmap(norm_c))
                    else:
                        C_a.append('white')
        if X_t:
            pause_quiver[0] = ax.quiver(
                X_t, Y_t, U_a, V_a,
                color=C_a,
                alpha=0.85,
                scale=22,          # smaller → longer arrows
                width=0.004,
                headwidth=4,
                headlength=5,
                zorder=5,
            )
    else:
        # Remove arrowheads when resuming
        if pause_quiver[0] is not None:
            pause_quiver[0].remove()
            pause_quiver[0] = None

    update_frame()

def on_regenerate(event):
    rebuild(sl_alpha.val, int(sl_n.val))

btn_pause.on_clicked(toggle_pause)
btn_regen.on_clicked(on_regenerate)

plt.show()
