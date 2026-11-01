import numpy as np
import pandas as pd
from pathlib import Path

# Locate files in your GitHub repository
REPO_DIR = Path(__file__).resolve().parents[3]

INPUT_CSV = REPO_DIR / "data" / "running_3d_coordinates csv.csv"

OUTPUT_CSV = (
    REPO_DIR / "research" / "running_kinematics"
    / "data" / "lower_limb_joint_angles.csv"
)


def angle_between_vectors(vector1, vector2):
    """Return the unsigned 3D angle between two vectors."""
    magnitude = (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    if magnitude < 1e-10:
        return np.nan

    cosine = np.clip(
        np.dot(vector1, vector2) / magnitude,
        -1.0,
        1.0
    )

    return np.degrees(np.arccos(cosine))


def calculate_knee_angle(hip, knee, ankle):
    """Estimated knee flexion: straight knee = 0 degrees."""
    thigh = hip - knee
    shank = ankle - knee

    internal_angle = angle_between_vectors(thigh, shank)

    return 180.0 - internal_angle


def calculate_hip_angle(pelvis, thorax, hip, knee):
    """
    Estimated hip angle between trunk and thigh.
    Neutral alignment is approximately 0 degrees.
    This is an unsigned 3D angle, not anatomical
    sagittal-plane hip flexion/extension.
    """
    trunk = thorax - pelvis
    thigh = knee - hip

    return 180.0 - angle_between_vectors(trunk, thigh)


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame", "time_seconds", "joint_index", "x", "y", "z"
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required CSV columns.")

    results = []

    for frame, group in df.groupby("frame", sort=True):

        if group["joint_index"].duplicated().any():
            raise ValueError(
                f"Duplicate joint indices in frame {frame}"
            )

        joints = group.set_index("joint_index")

        required_joints = [0, 1, 2, 3, 4, 5, 6, 8]

        if not all(j in joints.index for j in required_joints):
            raise ValueError(
                f"Missing required joints in frame {frame}"
            )

        def position(index):
            return joints.loc[
                index, ["x", "y", "z"]
            ].to_numpy(dtype=float)

        pelvis = position(0)
        thorax = position(8)

        right_hip = position(1)
        right_knee = position(2)
        right_ankle = position(3)

        left_hip = position(4)
        left_knee = position(5)
        left_ankle = position(6)

        results.append({
            "frame": frame,
            "time_seconds": group["time_seconds"].iloc[0],

            "right_hip_angle_deg": calculate_hip_angle(
                pelvis, thorax, right_hip, right_knee
            ),

            "left_hip_angle_deg": calculate_hip_angle(
                pelvis, thorax, left_hip, left_knee
            ),

            "right_knee_flexion_deg": calculate_knee_angle(
                right_hip, right_knee, right_ankle
            ),

            "left_knee_flexion_deg": calculate_knee_angle(
                left_hip, left_knee, left_ankle
            )
        })

    output = pd.DataFrame(results).sort_values("frame")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("=== Lower Limb Joint Angle Results ===")
    print(f"Frames analyzed: {len(output)}")

    print("\nFirst 10 frames:")
    print(output.head(10).to_string(index=False))

    print("\nAngle summary:")
    print(
        output[
            [
                "right_hip_angle_deg",
                "left_hip_angle_deg",
                "right_knee_flexion_deg",
                "left_knee_flexion_deg"
            ]
        ].describe().to_string()
    )

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
