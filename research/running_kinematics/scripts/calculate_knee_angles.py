import numpy as np
import pandas as pd
from pathlib import Path

# Locate the CSV in your GitHub repository
REPO_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = REPO_DIR / "data" / "running_3d_coordinates csv.csv"

# Save the results in your research folder
OUTPUT_CSV = (
    REPO_DIR / "research" / "running_kinematics"
    / "data" / "knee_angles.csv"
)


def calculate_knee_flexion(hip, knee, ankle):
    """
    Calculate estimated 3D knee flexion in degrees.
    A straight knee is approximately 0 degrees.
    """

    thigh = hip - knee
    shank = ankle - knee

    dot_product = np.dot(thigh, shank)
    magnitude = np.linalg.norm(thigh) * np.linalg.norm(shank)

    if magnitude < 1e-10:
        return np.nan

    cosine = np.clip(dot_product / magnitude, -1.0, 1.0)
    internal_angle = np.degrees(np.arccos(cosine))

    return 180.0 - internal_angle


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"frame", "time_seconds", "joint_index", "x", "y", "z"}

    if not required.issubset(df.columns):
        raise ValueError("The CSV is missing required columns.")

    results = []

    for frame, group in df.groupby("frame"):
        if group["joint_index"].duplicated().any():
            raise ValueError(f"Duplicate joint indices in frame {frame}")

        joints = group.set_index("joint_index")

        required_joints = [1, 2, 3, 4, 5, 6]
        if not all(joint in joints.index for joint in required_joints):
            raise ValueError(f"Missing knee landmarks in frame {frame}")

        def position(index):
            return joints.loc[index, ["x", "y", "z"]].to_numpy(
                dtype=float
            )

        right_knee = calculate_knee_flexion(
            position(1), position(2), position(3)
        )

        left_knee = calculate_knee_flexion(
            position(4), position(5), position(6)
        )

        results.append({
            "frame": frame,
            "time_seconds": group["time_seconds"].iloc[0],
            "right_knee_flexion_deg": right_knee,
            "left_knee_flexion_deg": left_knee,
        })

    output = pd.DataFrame(results).sort_values("frame")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("=== Knee Angle Results ===")
    print(f"Frames analyzed: {len(output)}")
    print(output.head(10).to_string(index=False))

    print("\nSummary:")
    print(
        output[
            ["right_knee_flexion_deg", "left_knee_flexion_deg"]
        ].describe().to_string()
    )

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
