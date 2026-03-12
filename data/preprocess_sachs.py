"""
Preprocess Sachs observational dataset.

Applies two transformations:
1. Remove outliers beyond 3 standard deviations from the mean in any variable.
2. Apply log transformation to reduce skewness and stabilize variance.

Outputs:
- sachs/sachs_observational_preprocessed.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def preprocess_sachs(
    input_path: str = "sachs/sachs_observational.csv",
    output_path: str = "sachs/sachs_observational_preprocessed.csv",
    std_threshold: float = 3.0,
) -> pd.DataFrame:
    data_dir = Path(__file__).parent
    df = pd.read_csv(data_dir / input_path, sep="\t")
    n_before = len(df)

    # 1. Remove outliers (if at least one variable > std_threshold standard deviations from the mean)
    """
    mean = df.mean()
    std = df.std()
    is_within_bounds = (df - mean).abs() <= std_threshold * std
    mask = is_within_bounds.all(axis=1)
    df = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"Outlier removal: {n_removed} rows removed ({n_before} -> {len(df)})")
    """

    # 2. Log transformation
    df = np.log(df)
    print("Log transformation applied.")

    # Save
    out = data_dir / output_path
    df.to_csv(out, sep="\t", index=False)
    print(f"Saved to: {out}")

    return df


if __name__ == "__main__":
    #preprocess_sachs()
    preprocess_sachs(output_path = "sachs/log_sachs_observational.csv")
