#!/usr/bin/env python3
"""
csv_to_input_package.py
=======================
Convert UMAP coordinate and label CSV files to the parquet format expected by
umap_viewer, and write a ready-to-use config.yaml.

Expected CSV schemas
--------------------
Coords CSV  (--coords):
    id      string   — unique point identifier
    x       float    — UMAP x coordinate
    y       float    — UMAP y coordinate

Labels CSV  (--labels, one per label set):
    id      string   — must match ids in the coords file
    labels  string   — category / cluster name for this point

Usage examples
--------------
Single label set:

    python csv_to_input_package.py \\
        --coords my_coords.csv \\
        --labels my_labels.csv \\
        --output-dir data

Multiple label sets with custom display names:

    python csv_to_input_package.py \\
        --coords coords.csv \\
        --labels layer0.csv layer1.csv layer2.csv \\
        --label-names "Layer 0" "Layer 1" "Layer 2" \\
        --output-dir data \\
        --config config.yaml

After running this script:
    Native viewer:   ./umap_viewer --config config.yaml
    Export for WASM: ./umap_viewer --config config.yaml --export-bin
                     ./webapp_build.sh
"""

import argparse
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError:
    sys.exit("polars is required:  pip install polars")

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required:  pip install pyyaml")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def load_coords(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    missing = {"id", "x", "y"} - set(df.columns)
    if missing:
        sys.exit(
            f"Coords CSV '{path}' is missing required columns: {missing}\n"
            f"  Found: {df.columns}"
        )
    df = df.select([
        pl.col("id").cast(pl.Utf8),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
    ])
    n_dupes = df.height - df["id"].n_unique()
    if n_dupes:
        print(
            f"  Warning: {n_dupes} duplicate id(s) in coords — keeping first occurrence",
            file=sys.stderr,
        )
        df = df.unique(subset=["id"], keep="first")
    return df


def load_labels(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    missing = {"id", "labels"} - set(df.columns)
    if missing:
        sys.exit(
            f"Labels CSV '{path}' is missing required columns: {missing}\n"
            f"  Found: {df.columns}"
        )
    return df.select([
        pl.col("id").cast(pl.Utf8),
        pl.col("labels").cast(pl.Utf8),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert UMAP coords + label CSVs to parquet and generate config.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coords", required=True, metavar="CSV",
        help="Coords CSV with columns: id, x, y",
    )
    parser.add_argument(
        "--labels", required=True, nargs="+", metavar="CSV",
        help="One or more label CSVs, each with columns: id, labels",
    )
    parser.add_argument(
        "--label-names", nargs="+", metavar="NAME",
        help=(
            "Display name for each label set shown in the viewer "
            "(default: CSV filename stem). Count must match --labels."
        ),
    )
    parser.add_argument(
        "--output-dir", default="data", metavar="DIR",
        help="Directory to write parquet files (default: data)",
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="FILE",
        help="Path for the generated config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output-bin", default="data/points.bin", metavar="FILE",
        help=(
            "output_bin path written into config.yaml — used by --export-bin "
            "when building the WASM version (default: data/points.bin)"
        ),
    )
    args = parser.parse_args()

    if args.label_names and len(args.label_names) != len(args.labels):
        sys.exit(
            f"--label-names count ({len(args.label_names)}) must match "
            f"--labels count ({len(args.labels)})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- coords -------------------------------------------------------
    coords_csv = Path(args.coords)
    print(f"Loading coords:  {coords_csv}")
    coords_df = load_coords(coords_csv)
    coords_parquet = output_dir / (coords_csv.stem + ".parquet")
    coords_df.write_parquet(coords_parquet)
    print(f"  {coords_df.height} points → {coords_parquet}")

    coords_ids = set(coords_df["id"].to_list())

    # ---- label sets ---------------------------------------------------
    label_entries: dict[str, str] = {}
    for i, label_csv_str in enumerate(args.labels):
        label_csv = Path(label_csv_str)
        name = args.label_names[i] if args.label_names else label_csv.stem

        print(f"Loading labels '{name}':  {label_csv}")
        labels_df = load_labels(label_csv)

        unmatched = set(labels_df["id"].to_list()) - coords_ids
        if unmatched:
            print(
                f"  Warning: {len(unmatched)} label id(s) not found in coords "
                f"(will be ignored during join in the viewer)",
                file=sys.stderr,
            )

        labels_parquet = output_dir / (label_csv.stem + ".parquet")
        labels_df.write_parquet(labels_parquet)
        print(f"  {labels_df.height} rows → {labels_parquet}")
        label_entries[name] = str(labels_parquet)

    # ---- config.yaml --------------------------------------------------
    config = {
        "coords_parquet": str(coords_parquet),
        "labels_parquet": label_entries,
        "output_bin": args.output_bin,
    }
    config_path = Path(args.config)
    with config_path.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"\nWrote {config_path}")

    print(
        f"\nNext steps:\n"
        f"  Native viewer : ./umap_viewer --config {config_path}\n"
        f"  Build for WASM: ./umap_viewer --config {config_path} --export-bin\n"
        f"                  ./webapp_build.sh"
    )


if __name__ == "__main__":
    main()
