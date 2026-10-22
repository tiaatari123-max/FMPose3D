import pandas as pd
from pathlib import Path

# This script is stored in research/running_kinematics/scripts/
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_CSV = DATA_DIR / "running_3d_coordinates_named.csv"
OUTPUT_CSV = DATA_DIR / "coordinate_quality_check.csv"


def main():
    print("=== Running Kinematics Coordinate Check ===")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find the coordinate CSV at: {INPUT_CSV}\n"
            "Please upload running_3d_coordinates_named.csv "
            "to research/running_kinematics/data/ on GitHub."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "frame",
        "time_seconds",
        "joint_index",
        "joint_name",
        "x",
        "y",
        "z",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The CSV is missing these columns: {missing_columns}"
        )

    joints_per_frame = df.groupby("frame")["joint_index"].nunique()

    print(f"Total rows: {len(df)}")
    print(f"Total frames: {df['frame'].nunique()}")
    print(f"Unique joint indices: {df['joint_index'].nunique()}")
    print(
        "Missing coordinate values: "
        f"{df[['x', 'y', 'z']].isna().sum().sum()}"
    )
    print(f"Minimum joints per frame: {joints_per_frame.min()}")
    print(f"Maximum joints per frame: {joints_per_frame.max()}")

    print("\nJoint indices and labels:")
    print(
        df[["joint_index", "joint_name"]]
        .drop_duplicates()
        .sort_values("joint_index")
        .to_string(index=False)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    report = df.groupby("frame").agg(
        time_seconds=("time_seconds", "first"),
        joint_count=("joint_index", "nunique"),
        missing_x=("x", lambda values: values.isna().sum()),
        missing_y=("y", lambda values: values.isna().sum()),
        missing_z=("z", lambda values: values.isna().sum()),
    ).reset_index()

    report.to_csv(OUTPUT_CSV, index=False)

    print(f"\nQuality report saved to: {OUTPUT_CSV}")
    print("Reminder: verify joint labels before calculating angles.")


if __name__ == "__main__":
    main()
