import numpy as np
import pandas as pd
from pathlib import Path

# Input: named 3D coordinates exported by FMPose3D.
# Update this path if your CSV is stored somewhere else.
INPUT_CSV = "../data/running_3d_coordinates_named.csv"

# Output file for data checks.
OUTPUT_CSV = "../data/coordinate_quality_check.csv"


def main():
    csv_path = Path(INPUT_CSV)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Check that the CSV is in the project's data folder."
        )

    df = pd.read_csv(csv_path)

    required = {
        "frame", "time_seconds", "joint_index",
        "joint_name", "x", "y", "z"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    print("=== FMPose3D coordinate data check ===")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['frame'].nunique()}")
    print(f"Unique joint indices: {df['joint_index'].nunique()}")
    print(f"Missing coordinate values: {df[['x', 'y', 'z']].isna().sum().sum()}")

    joints_per_frame = df.groupby("frame")["joint_index"].nunique()
    print(f"Minimum joints in a frame: {joints_per_frame.min()}")
    print(f"Maximum joints in a frame: {joints_per_frame.max()}")

    print("\nJoint indices and current labels:")
    print(
        df[["joint_index", "joint_name"]]
        .drop_duplicates()
        .sort_values("joint_index")
        .to_string(index=False)
    )

    print(
        "\nNote: joint labels and coordinate conventions must be verified "
        "before calculating biomechanical angles."
    )

    quality = df.groupby("frame").agg(
        time_seconds=("time_seconds", "first"),
        joint_count=("joint_index", "nunique"),
        missing_x=("x", lambda s: s.isna().sum()),
        missing_y=("y", lambda s: s.isna().sum()),
        missing_z=("z", lambda s: s.isna().sum()),
    ).reset_index()

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output_path, index=False)

    print(f"\nSaved frame-level quality report to: {output_path}")


if __name__ == "__main__":
    main()
