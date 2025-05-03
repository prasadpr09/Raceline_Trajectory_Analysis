import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import pandas as pd
import cv2

# === CONFIG ===
map_img_path = 'race3.pgm'
map_yaml_path = 'race3.yaml'
raceline_path = 'optimals/optimal2.csv'
save_path = 'optimals/overtaking.csv'
downsample_rate = 200  # edit every Nth point
density_factor = 1  # save N times denser

# === Load Map YAML ===
with open(map_yaml_path, 'r') as f:
    map_metadata = yaml.safe_load(f)

resolution = map_metadata['resolution']
origin = np.array(map_metadata['origin'][:2])
negate = map_metadata.get('negate', 0)

# === Load Map Image ===
map_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
if negate:
    map_img = 255 - map_img
map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_GRAY2RGB)

# === Coordinate Conversions ===
def world_to_pixel(pts):
    px = ((pts[:, 0] - origin[0]) / resolution).astype(np.int32)
    py = (map_img.shape[0] - ((pts[:, 1] - origin[1]) / resolution)).astype(np.int32)
    return np.stack((px, py), axis=1)

def pixel_to_world(pix_pts):
    x = pix_pts[:, 0] * resolution + origin[0]
    y = (map_img.shape[0] - pix_pts[:, 1]) * resolution + origin[1]
    return np.stack((x, y), axis=1)

# === Load Raceline CSV ===
full_raceline_df = pd.read_csv(raceline_path, comment='#', delimiter=';', header=None)
# full_raceline_df.columns = [
#     's_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm',
#     'vx_mps', 'ax_mps2', 'ay_mps2', 'omega_radps', 't_sec'
# ][:full_raceline_df.shape[1]]
# full_raceline_xy = full_raceline_df[['x_m', 'y_m']].to_numpy()
# full_raceline_vx = full_raceline_df['vx_mps'].to_numpy()


full_raceline_df.columns = [
    'x','y','speed','lookahead'
][:full_raceline_df.shape[1]]

# Extract x, y, and yaw
# x = full_raceline_df['x'].to_numpy()
# y = full_raceline_df['y'].to_numpy()
full_raceline_xy = full_raceline_df[['x', 'y']].to_numpy()
full_raceline_vx = full_raceline_df['speed'].to_numpy()



# === Downsample for Editing ===
sparse_idx = np.arange(0, len(full_raceline_xy), downsample_rate)
sparse_raceline_xy = full_raceline_xy[sparse_idx]
sparse_raceline_vx = full_raceline_vx[sparse_idx]
sparse_raceline_px = world_to_pixel(sparse_raceline_xy)

# === Plotting and GUI ===
fig, ax = plt.subplots()
plt.imshow(map_img_rgb)
line_plot, = ax.plot(sparse_raceline_px[:, 0], sparse_raceline_px[:, 1], 'r-', lw=2)
pts_plot = ax.scatter(sparse_raceline_px[:, 0], sparse_raceline_px[:, 1], c='b', s=10)

selected_idx = None
velocity_text = None

# === Events ===
def update_curve(local_k=3):
    global sparse_raceline_px, selected_idx
    if selected_idx is None:
        return

    start = max(0, selected_idx - local_k)
    end = min(len(sparse_raceline_px), selected_idx + local_k + 1)

    local_pts = sparse_raceline_px[start:end]
    if len(local_pts) >= 3:
        t = np.linspace(0, 1, len(local_pts))
        x = np.interp(t, t, local_pts[:, 0])
        y = np.interp(t, t, local_pts[:, 1])
        sparse_raceline_px[start:end] = np.stack((x, y), axis=1).astype(np.int32)

    line_plot.set_data(sparse_raceline_px[:, 0], sparse_raceline_px[:, 1])
    pts_plot.set_offsets(sparse_raceline_px)
    fig.canvas.draw_idle()

def update_velocity(local_k=3):
    global sparse_raceline_vx, selected_idx
    if selected_idx is None:
        return

    start = max(0, selected_idx - local_k)
    end = min(len(sparse_raceline_vx), selected_idx + local_k + 1)

    local_vx = sparse_raceline_vx[start:end]
    if len(local_vx) >= 3:
        t = np.linspace(0, 1, len(local_vx))
        smooth_vx = np.interp(t, t, local_vx)
        sparse_raceline_vx[start:end] = smooth_vx

def on_click(event):
    global selected_idx, velocity_text
    if event.inaxes != ax:
        return
    mouse = np.array([event.xdata, event.ydata])
    dists = np.linalg.norm(sparse_raceline_px - mouse, axis=1)
    if np.min(dists) < 15:
        selected_idx = np.argmin(dists)

        # Update velocity display
        if velocity_text is not None:
            velocity_text.remove()
        velocity_text = ax.text(
            sparse_raceline_px[selected_idx, 0],
            sparse_raceline_px[selected_idx, 1] - 20,
            f"v={sparse_raceline_vx[selected_idx]:.2f} m/s",
            color='yellow', fontsize=10, weight='bold'
        )
        fig.canvas.draw_idle()

def on_release(event):
    global selected_idx
    if selected_idx is not None:
        update_curve()
        selected_idx = None

def on_motion(event):
    if selected_idx is not None and event.inaxes == ax:
        sparse_raceline_px[selected_idx] = [event.xdata, event.ydata]
        update_curve()

def on_key(event):
    global selected_idx, velocity_text
    if selected_idx is None:
        return

    if event.key in  ['+', '=']:
        sparse_raceline_vx[selected_idx] += 0.1
        update_velocity()
    elif event.key == '-':
        sparse_raceline_vx[selected_idx] -= 0.1
        update_velocity()

    # Update velocity text
    if velocity_text is not None:
        velocity_text.set_text(f"v={sparse_raceline_vx[selected_idx]:.2f} m/s")
    print(f"Velocity at selected point: {sparse_raceline_vx[selected_idx]:.2f} m/s")
    fig.canvas.draw_idle()

# def save_callback(event):
#     updated_sparse_world = pixel_to_world(sparse_raceline_px)

#     sparse_indices = sparse_idx
#     full_indices = np.arange(len(full_raceline_df))
#     new_dense_indices = np.linspace(0, len(full_raceline_df) - 1, len(full_raceline_df) * density_factor)

#     interp_x = interp1d(sparse_indices, updated_sparse_world[:, 0], kind='linear', fill_value="extrapolate")
#     interp_y = interp1d(sparse_indices, updated_sparse_world[:, 1], kind='linear', fill_value="extrapolate")
#     interp_vx = interp1d(sparse_indices, sparse_raceline_vx, kind='linear', fill_value="extrapolate")

#     new_x = interp_x(new_dense_indices)
#     new_y = interp_y(new_dense_indices)
#     new_vx = interp_vx(new_dense_indices)

#     updated_df = pd.DataFrame()
#     updated_df['x_m'] = new_x
#     updated_df['y_m'] = new_y

#     copy_columns = ['s_m', 'psi_rad', 'kappa_radpm', 'ax_mps2', 'ay_mps2', 'omega_radps', 't_sec']
#     for col in copy_columns:
#         if col in full_raceline_df.columns:
#             if col in ['s_m', 't_sec']:
#                 updated_df[col] = np.linspace(full_raceline_df[col].iloc[0], full_raceline_df[col].iloc[-1], len(new_dense_indices))
#             else:
#                 interp_col = interp1d(full_indices, full_raceline_df[col], kind='linear', fill_value="extrapolate")
#                 updated_df[col] = interp_col(new_dense_indices)

#     updated_df['vx_mps'] = new_vx

#     final_cols = [col for col in full_raceline_df.columns if col in updated_df.columns]
#     updated_df = updated_df[final_cols]

#     updated_df.to_csv(save_path, sep=';', index=False)
#     print(f"✅ Full dense raceline with updated velocity saved ({len(new_dense_indices)} points) to: {save_path}")


def save_callback(event):
    updated_sparse_world = pixel_to_world(sparse_raceline_px)

    sparse_indices = sparse_idx
    full_indices = np.arange(len(full_raceline_df))
    new_dense_indices = np.linspace(0, len(full_raceline_df) - 1, len(full_raceline_df) * density_factor)

    interp_x = interp1d(sparse_indices, updated_sparse_world[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(sparse_indices, updated_sparse_world[:, 1], kind='linear', fill_value="extrapolate")
    interp_vx = interp1d(sparse_indices, sparse_raceline_vx, kind='linear', fill_value="extrapolate")

    # # 🆕 Interpolate lookahead
    sparse_raceline_lookahead = full_raceline_df['lookahead'].to_numpy()[sparse_idx]
    interp_lookahead = interp1d(sparse_indices, sparse_raceline_lookahead, kind='linear', fill_value="extrapolate")

    new_x = interp_x(new_dense_indices)
    new_y = interp_y(new_dense_indices)
    new_vx = interp_vx(new_dense_indices)
    new_lookahead = interp_lookahead(new_dense_indices)

    # 🆕 Save lookahead too
    updated_df = pd.DataFrame({
        'x': new_x,
        'y': new_y,
        'speed': new_vx,
        'lookahead': new_lookahead
    })

    updated_df.to_csv(save_path, sep=';', index=False)
    print(f"✅ Full dense raceline with updated velocity and lookahead saved ({len(new_dense_indices)} points) to: {save_path}")




# === Hook Events ===
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('key_press_event', on_key)

# === Save Button ===
ax_save = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_save = Button(ax_save, 'Save')
btn_save.on_clicked(save_callback)

plt.title("Drag to Edit Path, +/- to Edit Velocity")
plt.show()