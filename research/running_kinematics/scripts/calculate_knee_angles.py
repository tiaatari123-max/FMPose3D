import pandas as pd
from pathlib import Path

# Locate the repository's root data folder.
REPO_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = REPO_DIR / "data" / "running_3d_coordinates csv.csv"


def main():
    print("=== Running Kinematics Coordinate Check ===")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"CSV file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("\nCSV column names:")
    print(df.columns.tolist())

    print(f"\nTotal rows: {len(df)}")
    print(f"Missing values by column:\n{df.isna().sum()}")

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print(
        "\nThis is a data check only. Joint angles will be calculated "
        "after we verify the joint labels and coordinate conventions."
    )


if __name__ == "__main__":
    main()
