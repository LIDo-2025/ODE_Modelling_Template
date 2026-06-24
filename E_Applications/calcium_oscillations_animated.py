from numpy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Model equations
def model(t, y, params):

    c1, c2, c3 = y
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = params

    if t > 100 and t < 200:

        k1 = 0.55
    
    dc1_dt = k1 + k2*c1 - k3*c1*c2/(k4 + c1) - k5*c1*c3/(k6 + c1)
    dc2_dt = k7*c1 - k8*c2/(k9 + c2)
    dc3_dt = k10*c1 - k11*c3/(k12 + c3)

    return [dc1_dt, dc2_dt, dc3_dt]

# Initial conditions and parameters

c1_0 = 0.556  # Initial value for c1 (replace with actual value)
c2_0 = 0.636  # Initial value for c2 (replace with actual value)
c3_0 = 0.0083  # Initial value for c2 (replace with actual value)

y0 = [c1_0, c2_0, c3_0]

params = [0.01, 1.3, 1.52, 0.19, 4.88, 1.18, 1.24, 32.24, 29.09, 13.58, 153, 0.16]

time_span = (0, 250)
# Solve the ODEs
solution = solve_ivp(model, time_span, y0, args=(params,), method='BDF', max_step=0.5)
t, traject_pos = solution.t, solution.y.T

# Set up the figure with 2 subplots (2 rows, 1 column)
fig = plt.figure(figsize=(8, 10))

ax1 = fig.add_subplot(211, projection='3d')  # 3D state space
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 3)
ax1.set_zlim(0, 0.06)
ax1.view_init(elev=30, azim=120)

ax2 = fig.add_subplot(212)                  # 2D time series

# Subplot 1: 3D trajectory (with tail)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Calcium')
ax1.set_title('3D State Space', fontsize=12)
traject_point, = ax1.plot([], [], [], 'ro', markersize=20)
trail, = ax1.plot([], [], [], 'b-', linewidth=2, alpha=0.5)
trail_length = 100

# Subplot 2: Time series of c1 (or c2/c3)
ax2.set_xlabel('Time'); ax2.set_ylabel('Calcium')
ax2.set_title('Cytosolic Calcium', fontsize=12)
ax2.set_ylim(0, 0.07)
time_line, = ax2.plot([], [], 'b-', label='c3(t)')
current_time_marker, = ax2.plot([], [], 'ro', markersize=8, label='Current time')

# Initialize both subplots
def init():
    # 3D plot
    traject_point.set_data([], [])
    traject_point.set_3d_properties([])
    trail.set_data([], [])
    trail.set_3d_properties([])
    
    # Time series plot
    time_line.set_data([], [])
    current_time_marker.set_data([], [])  # Initialize with empty lists
    
    return [traject_point, trail, time_line, current_time_marker]

def update(frame):
    frame = min(frame, len(traject_pos) - 1)
    
    # --- Update 3D plot ---
    x, y, z = traject_pos[frame, 0], traject_pos[frame, 1], traject_pos[frame, 2]
    traject_point.set_data([x], [y])
    traject_point.set_3d_properties([z])
    
    # Update tail
    start = max(0, frame - trail_length)
    trail_x = traject_pos[start:frame+1, 0]
    trail_y = traject_pos[start:frame+1, 1]
    trail_z = traject_pos[start:frame+1, 2]
    trail.set_data(trail_x, trail_y)
    trail.set_3d_properties(trail_z)
    
    # --- Update time series plot ---
    time_line.set_data(t[:frame+1], traject_pos[:frame+1, 2])  # Plot c3 vs time
    
    # Fix: Pass lists/arrays to set_data
    current_time_marker.set_data([t[frame]], [traject_pos[frame, 2]])  # Red dot at current time
    
    # Adjust time series axis limits dynamically (optional)
    ax2.relim()
    ax2.autoscale_view()
    
    return [traject_point, trail, time_line, current_time_marker]

# Animate
ani = FuncAnimation(fig, update, frames=len(traject_pos),
                   init_func=init, blit=False, interval=25)

# ani.save('cytosolic_calcium_oscillation.mp4', writer='ffmpeg', fps=30, dpi=200)

plt.tight_layout()
plt.show()