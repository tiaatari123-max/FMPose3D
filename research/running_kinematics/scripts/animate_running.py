import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Locate files in your GitHub repository
REPO_DIR = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    REPO_DIR / "data" / "running_3d_coordinates csv.csv"
)

OUTPUT_DIR = (
    REPO_DIR / "research" / "running_kinematics" / "data"
)

OUTPUT_GIF = OUTPUT_DIR / "running_3d_animation.gif"

# Human3.6M skeleton connections
SKELETON = [
    (0, 1), (1, 2), (2, 3),       # Right leg
    (0, 4), (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8), (8, 9),       # Torso and neck
    (9, 10),                     # Head
    (8, 11), (11, 12), (12, 13), # Left arm
    (8, 14), (14, 15), (15, 16)  # Right arm
]

# Read 3D coordinates
df = pd.read_csv(INPUT_CSV)

required = {"frame", "joint_index", "x", "y", "z"}
if not required.issubset(df.columns):
    raise ValueError("Missing required coordinate columns.")

frames = []

for frame_number, group in df.groupby("frame", sort=True):
    if len(group) != 17 or group["joint_index"].duplicated().any():
        raise ValueError(f"Invalid joints in frame {frame_number}")

    group = group.set_index("joint_index").reindex(range(17))

    if group[["x", "y", "z"]].isna().any().any():
        raise ValueError(f"Missing coordinates in frame {frame_number}")

    frames.append(
        group[["x", "y", "z"]].to_numpy(dtype=float)
    )

poses = np.stack(frames)

# Center the skeleton at the pelvis in every frame
poses = poses - poses[:, 0:1, :]

# Consistent axis limits throughout the animation
limit = max(0.5, float(np.max(np.abs(poses))) * 1.1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

lines = []

for start, end in SKELETON:
    line, = ax.plot([], [], [], linewidth=2.5)
    lines.append(line)

joints = ax.scatter([], [], [], s=20)

def update(frame_index):
    pose = poses[frame_index]

    for line, (start, end) in zip(lines, SKELETON):
        segment = pose[[start, end]]

        line.set_data(segment[:, 0], segment[:, 1])
        line.set_3d_properties(segment[:, 2])

    joints._offsets3d = (
        pose[:, 0], pose[:, 1], pose[:, 2]
    )

    ax.set_title(
        f"Estimated 3D Running Pose | Frame {frame_index + 1}"
    )

    return lines + [joints]

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=15, azim=-70)

animation = FuncAnimation(
    fig,
    update,
    frames=len(poses),
    interval=1000 / 29.8876,
    blit=False
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

animation.save(
    OUTPUT_GIF,
    writer=PillowWriter(fps=15),
    dpi=90
)

plt.close(fig)

print(f"Successfully animated {len(poses)} frames.")
print(f"Animation saved to: {OUTPUT_GIF}")
