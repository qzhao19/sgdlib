#!/bin/bash
# # Script to prepare SUSY dataset for sgdlib benchmark
# # Downloads, extracts and converts label format (0 -> -1)

set -euo pipefail

# Configuration
readonly DATA_URL="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TARGET_DIR="${PROJECT_ROOT}/dataset"
readonly OUTPUT_FILE="${TARGET_DIR}/SUSY_processed"

# Initialize directory
echo "Creating target directory: ${TARGET_DIR}"
mkdir -p "$TARGET_DIR" || {
    echo "ERROR: Failed to create directory ${TARGET_DIR}" >&2
    exit 1
}

cd "$TARGET_DIR" || {
    echo "ERROR: Failed to enter directory ${TARGET_DIR}" >&2
    exit 1
}

# Download with retry and progress
for i in {1..3}; do
    if wget -q --show-progress "$DATA_URL" -O "SUSY.xz"; then
        break
    elif [[ $i -eq 3 ]]; then
        echo "ERROR: Download failed after 3 attempts" >&2
        exit 1
    fi
    sleep 5
done

# Process data
echo "Processing SUSY dataset..."
xz -dk "SUSY.xz" || {
    echo "ERROR: Failed to decompress SUSY.xz" >&2
    exit 1
}

[ -f "$TARGET_DIR/SUSY" ] || {
    echo "ERROR: Decompressed file SUSY not found" >&2
    exit 1
}

awk 'BEGIN{OFS="\t"} $1==0{$1=-1} {print}' "SUSY" > "susy_processed" || {
    echo "ERROR: Data processing failed" >&2
    exit 1
}

# Verification data
echo -e "\nVerification:"
echo "Original lines: $(wc -l < "SUSY")"
echo "Processed lines: $(wc -l < "susy_processed")"
echo "Output saved to: $(realpath "susy_processed")"

# cleanup
rm -f "$TARGET_DIR/SUSY.xz" 