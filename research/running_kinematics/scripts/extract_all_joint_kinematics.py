import numpy as np
import pandas as pd
from pathlib import Path

# File locations
REPO_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = REPO_DIR / "data" / "running_3d_coordinates csv.csv"
OUTPUT_DIR = REPO_DIR / "research" / "running_kinematics" / "data"

# Human3.6M joint names
JOINT_NAMES = {
    0: "Pelvis",
    1: "Right Hip",
    2: "Right Knee",
    3: "Right Ankle",
    4: "Left Hip",
    5: "Left Knee",
    6: "Left Ankle",
    7: "Spine",
    8: "Thorax",
    9: "Neck",
    10: "Head",
    11: "Left Shoulder",
    12: "Left Elbow",
    13: "Left Wrist",
    14: "Right Shoulder",
    15: "Right Elbow",
    16: "Right Wrist",
}


def vector_angle(a, b):
    """Unsigned 3D angle between vectors, in degrees."""
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)

    if not np.isfinite(magnitude) or magnitude < 1e-10:
        return np.nan

    cosine = np.clip(np.dot(a, b) / magnitude, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def joint_angle(a, vertex, b):
    """Internal 3D angle at the middle joint."""
    return vector_angle(a - vertex, b - vertex)


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame", "time_seconds", "joint_index", "x", "y", "z"
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required coordinate columns.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values in the coordinate data.")

    if not np.isfinite(df[["x", "y", "z"]].to_numpy()).all():
        raise ValueError("Non-finite coordinates found.")

    if df.duplicated(["frame", "joint_index"]).any():
        raise ValueError("Duplicate joints found within a frame.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # FILE 1: All 17 labeled joint coordinates
    coordinates = df.copy()
    coordinates["joint_name"] = coordinates["joint_index"].map(
        JOINT_NAMES
    )

    if coordinates["joint_name"].isna().any():
        raise ValueError("Unknown joint index found.")

    coordinates = coordinates[
        [
            "frame", "time_seconds", "joint_index",
            "joint_name", "x", "y", "z"
        ]
    ].sort_values(["frame", "joint_index"])

    coordinates.to_csv(
        OUTPUT_DIR / "all_17_joint_coordinates.csv",
        index=False
    )

    # FILE 2: Calculable 3D joint and segment angles
    results = []

    for frame, group in coordinates.groupby("frame", sort=True):
        if set(group["joint_index"]) != set(range(17)):
            raise ValueError(
                f"Frame {frame} does not contain all 17 joints."
            )

        if group["time_seconds"].nunique() != 1:
            raise ValueError(
                f"Inconsistent timestamps in frame {frame}."
            )

        joints = group.set_index("joint_index")

        def p(index):
            return joints.loc[
                index, ["x", "y", "z"]
            ].to_numpy(dtype=float)

        pelvis = p(0)
        right_hip, right_knee, right_ankle = p(1), p(2), p(3)
        left_hip, left_knee, left_ankle = p(4), p(5), p(6)

        spine, thorax, neck, head = p(7), p(8), p(9), p(10)

        left_shoulder, left_elbow, left_wrist = (
            p(11), p(12), p(13)
        )
        right_shoulder, right_elbow, right_wrist = (
            p(14), p(15), p(16)
        )

        # Trunk direction, pointing upward
        trunk = thorax - pelvis

        # Hip and knee flexion-related proxies:
        # 0 degrees represents approximately straight alignment.
        right_hip_angle = 180 - vector_angle(
            trunk, right_knee - right_hip
        )
        left_hip_angle = 180 - vector_angle(
            trunk, left_knee - left_hip
        )

        right_knee_angle = 180 - joint_angle(
            right_hip, right_knee, right_ankle
        )
        left_knee_angle = 180 - joint_angle(
            left_hip, left_knee, left_ankle
        )

        # Elbow flexion proxies
        right_elbow_angle = 180 - joint_angle(
            right_shoulder, right_elbow, right_wrist
        )
        left_elbow_angle = 180 - joint_angle(
            left_shoulder, left_elbow, left_wrist
        )

        # Shoulder internal 3D segment angles
        right_shoulder_angle = joint_angle(
            thorax, right_shoulder, right_elbow
        )
        left_shoulder_angle = joint_angle(
            thorax, left_shoulder, left_elbow
        )

        # Additional spinal and neck segment angles
        spine_angle = joint_angle(pelvis, spine, thorax)
        neck_angle = joint_angle(thorax, neck, head)

        results.append({
            "frame": frame,
            "time_seconds": group["time_seconds"].iloc[0],
            "right_hip_angle_deg": right_hip_angle,
            "left_hip_angle_deg": left_hip_angle,
            "right_knee_flexion_deg": right_knee_angle,
            "left_knee_flexion_deg": left_knee_angle,
            "right_elbow_flexion_deg": right_elbow_angle,
            "left_elbow_flexion_deg": left_elbow_angle,
            "right_shoulder_internal_angle_deg": right_shoulder_angle,
            "left_shoulder_internal_angle_deg": left_shoulder_angle,
            "spine_internal_angle_deg": spine_angle,
            "neck_internal_angle_deg": neck_angle,
        })

    angles = pd.DataFrame(results).sort_values("frame")

    angles.to_csv(
        OUTPUT_DIR / "all_joint_angles.csv",
        index=False
    )

    print("=== 3D RUNNING KINEMATICS ===")
    print(f"Frames analyzed: {len(angles)}")
    print(f"Joint coordinate records: {len(coordinates)}")
    print(f"Angle variables: {len(angles.columns) - 2}")

    print("\nFirst five frames:")
    print(angles.head().to_string(index=False))

    print("\nAngle summary:")
    print(angles.drop(columns=["frame", "time_seconds"])
          .describe().to_string())

    print("\nFiles saved:")
    print(OUTPUT_DIR / "all_17_joint_coordinates.csv")
    print(OUTPUT_DIR / "all_joint_angles.csv")


if __name__ == "__main__":
    main()
