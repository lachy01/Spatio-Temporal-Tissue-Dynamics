#!/bin/bash
# Download the two Xenium files MisTIC needs that the rest of the project does
# not: per-transcript detections and cell boundary polygons (1.7 GB together),
# from the same 10x location as cells.parquet and cell_feature_matrix.h5.
#
#   bash hpc/mistic_fetch.sh
#
# Run on a login node (compute nodes may have no internet). Skips files already
# present.
set -euo pipefail
source "$(dirname "$0")/config.sh"
DIR="$MIX_ROOT/data/Xenium_Prime_Breast_Cancer_FFPE"
URL="https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE"
mkdir -p "$DIR"
for f in transcripts.parquet cell_boundaries.parquet cells.parquet cell_feature_matrix.h5; do
  if [ -s "$DIR/$f" ]; then
    echo "have $f"
  else
    curl -sSfL -o "$DIR/$f.part" "${URL}_$f" && mv "$DIR/$f.part" "$DIR/$f"
    echo "fetched $f ($(du -h "$DIR/$f" | cut -f1))"
  fi
done
