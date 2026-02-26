"""
Download Sachs protein signaling dataset.

This script downloads the famous Sachs et al. (2005) protein signaling dataset
from the bnlearn repository. The dataset contains measurements of 11 proteins
and phospholipids from flow cytometry experiments on human immune cells.

Reference:
Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005).
Causal protein-signaling networks derived from multiparameter single-cell data.
Science, 308(5721), 523-529.
https://www.cs.columbia.edu/~dpeer/pub/science2005.pdf

Dataset sources:
- bnlearn: https://www.bnlearn.com/research/sachs05/
- Zenodo: https://zenodo.org/records/7681811
"""

import urllib.request
import gzip
import shutil
from pathlib import Path
import pandas as pd


def download_sachs_dataset(output_dir: str = "data/sachs") -> Path:
    """
    Download the Sachs protein signaling dataset.

    Args:
        output_dir: Directory where to save the dataset

    Returns:
        Path to the downloaded CSV file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # URLs for the dataset files
    observational_url = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"
    interventional_url = "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz"

    print("=" * 70)
    print("Downloading Sachs Protein Signaling Dataset")
    print("=" * 70)
    print()

    # Download observational data
    print("1. Downloading observational data...")
    obs_gz_path = output_path / "sachs.observational.txt.gz"
    obs_txt_path = output_path / "sachs_observational.csv"

    urllib.request.urlretrieve(observational_url, obs_gz_path)
    print(f"   Downloaded to: {obs_gz_path}")

    # Decompress
    with gzip.open(obs_gz_path, 'rb') as f_in:
        with open(obs_txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"   Decompressed to: {obs_txt_path}")
    obs_gz_path.unlink()  # Remove .gz file
    print()

    # Download interventional data
    print("2. Downloading interventional data...")
    int_gz_path = output_path / "sachs.interventional.txt.gz"
    int_txt_path = output_path / "sachs_interventional.csv"

    urllib.request.urlretrieve(interventional_url, int_gz_path)
    print(f"   Downloaded to: {int_gz_path}")

    # Decompress
    with gzip.open(int_gz_path, 'rb') as f_in:
        with open(int_txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"   Decompressed to: {int_txt_path}")
    int_gz_path.unlink()  # Remove .gz file
    print()

    # Display dataset info
    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)
    print()

    # Read and display observational data info
    obs_data = pd.read_csv(obs_txt_path, sep="\t")
    print(f"Observational data: {obs_data.shape[0]} samples, {obs_data.shape[1]} variables")
    print(f"  Variables: {', '.join(obs_data.columns)}")
    print()

    # Read and display interventional data info
    int_data = pd.read_csv(int_txt_path, sep=" ", skipinitialspace=True, quotechar='"')
    print(f"Interventional data: {int_data.shape[0]} samples, {int_data.shape[1]} variables")
    print(f"  Variables: {', '.join(int_data.columns)}")

    if "INT" in int_data.columns:
        interventions = int_data["INT"].value_counts().to_dict()
        print(f"  Interventions: {interventions}")
    print()

    print("=" * 70)
    print("Download complete!")
    print("=" * 70)
    print()
    print(f"Files saved in: {output_path.absolute()}")
    print(f"  - {obs_txt_path.name} (observational)")
    print(f"  - {int_txt_path.name} (interventional)")
    print()

    # Create a README
    readme_path = output_path / "README.md"
    with open(readme_path, "w") as f:
        f.write("""# Sachs Protein Signaling Dataset

## Reference

Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005).
Causal protein-signaling networks derived from multiparameter single-cell data.
Science, 308(5721), 523-529.

DOI: 10.1126/science.1105809

## Description

This dataset contains measurements of 11 proteins and phospholipids from flow
cytometry experiments on human CD4+ T cells. The data includes both observational
measurements and interventional experiments (knockdowns and activations).

## Variables

The 11 measured proteins/phospholipids are:
- **praf**: Phosphorylated Raf
- **pmek**: Phosphorylated MEK
- **plcg**: Phospholipase C-gamma
- **PIP2**: Phosphatidylinositol 4,5-bisphosphate
- **PIP3**: Phosphatidylinositol 3,4,5-trisphosphate
- **p44.42**: Phosphorylated ERK (p44/42)
- **pakts473**: Phosphorylated AKT (Ser473)
- **PKA**: Protein kinase A
- **PKC**: Protein kinase C
- **P38**: p38 MAP kinase
- **pjnk**: Phosphorylated JNK

## Files

- `sachs_observational.csv`: Observational data (no interventions)
- `sachs_interventional.csv`: Data with interventions (includes INT column)

## Known Causal Structure

The true causal network is documented in the original paper. This makes it ideal
for benchmarking structure learning algorithms.

## Sources

- bnlearn: https://www.bnlearn.com/research/sachs05/
- Zenodo: https://zenodo.org/records/7681811
""")

    print(f"Created README: {readme_path}")
    print()

    # Download ground truth from Zenodo
    print("=" * 70)
    print("Downloading Ground Truth Structure")
    print("=" * 70)
    print()

    zenodo_zip_url = "https://zenodo.org/records/7681811/files/sachs.zip?download=1"
    zip_path = output_path / "sachs_zenodo.zip"

    print("Downloading Zenodo dataset (contains ground truth)...")
    urllib.request.urlretrieve(zenodo_zip_url, zip_path)
    print(f"   Downloaded to: {zip_path}")

    # Extract only the GroundTruth.csv file
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract('GroundTruth.csv', output_path)

    # Rename to be consistent
    zenodo_gt_path = output_path / "GroundTruth.csv"
    zenodo_gt_renamed = output_path / "sachs_ground_truth_zenodo.csv"
    zenodo_gt_path.rename(zenodo_gt_renamed)

    print(f"   Extracted ground truth to: {zenodo_gt_renamed}")
    zip_path.unlink()  # Clean up zip file
    print()

    # Display ground truth info
    gt_data = pd.read_csv(zenodo_gt_renamed)
    print(f"Ground truth structure: {len(gt_data)} edges")
    print()

    print("=" * 70)
    print("IMPORTANT: Ground Truth Versions")
    print("=" * 70)
    print()
    print("Two versions of the ground truth exist:")
    print("  1. Zenodo version (sachs_ground_truth_zenodo.csv): 20 edges")
    print("  2. bnlearn consensus version: 17 edges")
    print()
    print("The difference is due to additional edges in the Zenodo version.")
    print("Use the version that best matches your research needs.")
    print()

    print("=" * 70)
    print("Download complete!")
    print("=" * 70)
    print()
    print("Downloaded files:")
    print(f"  - {obs_txt_path.name} (observational data)")
    print(f"  - {int_txt_path.name} (interventional data)")
    print(f"  - {zenodo_gt_renamed.name} (ground truth 20 edges, Zenodo)")
    print()
    print("Note: sachs_ground_truth.csv (17 edges) is already in the repository.")
    print()

    return obs_txt_path


if __name__ == "__main__":
    download_sachs_dataset()
