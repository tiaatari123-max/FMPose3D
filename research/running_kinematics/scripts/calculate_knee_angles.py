import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# File locations in your GitHub repository
REPO_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = REPO_DIR / "data" / "running_3d_coordinates csv.csv"
OUTPUT_DIR = REPO_DIR / "research" / "running_kinematics" / "data"


def calculate_knee_flexion(hip, knee, ankle):
    """Calculate estimated 3D knee flexion in degrees."""

    thigh = hip - knee
    shank = ankle - knee

    magnitude = np.linalg.norm(thigh) * np.linalg.norm(shank)

    if magnitude < 1e-10:
        return np.nan

    cosine = np.clip(
        np.dot(thigh, shank) / magnitude,
        -1.0,
        1.0
    )

    internal_angle = np.degrees(np.arccos(cosine))

    return 180.0 - internal_angle


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"frame", "time_seconds", "joint_index", "x", "y", "z"}

    if not required.issubset(df.columns):
        raise ValueError("Missing required CSV columns.")

    results = []

    for frame, group in df.groupby("frame"):
        if group["joint_index"].duplicated().any():
            raise ValueError(f"Duplicate joint indices in frame {frame}")

        joints = group.set_index("joint_index")

        if not all(j in joints.index for j in [1, 2, 3, 4, 5, 6]):
            raise ValueError(f"Missing knee landmarks in frame {frame}")

        def position(index):
            return joints.loc[index, ["x", "y", "z"]].to_numpy(
                dtype=float
            )

        right_angle = calculate_knee_flexion(
            position(1), position(2), position(3)
        )

        left_angle = calculate_knee_flexion(
            position(4), position(5), position(6)
        )

        results.append({
            "frame": frame,
            "time_seconds": group["time_seconds"].iloc[0],
            "right_knee_flexion_deg": right_angle,
            "left_knee_flexion_deg": left_angle,
        })

    output = pd.DataFrame(results).sort_values("frame")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_DIR / "knee_angles.csv", index=False)

    # Generate separate graphs for each knee
    for side, color in [
        ("right", "royalblue"),
        ("left", "darkorange"),
    ]:
        column = f"{side}_knee_flexion_deg"

        plt.figure(figsize=(10, 5))

        plt.plot(
            output["time_seconds"],
            output[column],
            color=color,
            linewidth=1.5
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Estimated 3D Knee Flexion (degrees)")
        plt.title(f"{side.capitalize()} Knee Flexion During Running")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = OUTPUT_DIR / f"{side}_knee_flexion.png"
        plt.savefig(filename, dpi=200)
        plt.close()

        print(f"Saved graph: {filename}")

    print("\n=== Knee Angle Summary ===")
    print(f"Frames analyzed: {len(output)}")

    print(
        output[
            ["right_knee_flexion_deg", "left_knee_flexion_deg"]
        ].describe().to_string()
    )

    print("\nSaved knee angles and both graphs.")


if __name__ == "__main__":
    main()
