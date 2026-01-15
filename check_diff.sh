#!/usr/bin/env bash
# compare_dirs.sh
#
# Compare two directories (recursively) and show:
#  - files in DIR1 but missing in DIR2
#  - files in DIR2 but missing in DIR1
#
# Paths are shown *relative* to each directory root.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 DIR1 DIR2" >&2
    echo "Example: $0 /path/to/ref_dir /path/to/other_dir" >&2
    exit 1
fi

DIR1=$1
DIR2=$2

# Check that both are directories
for d in "$DIR1" "$DIR2"; do
    if [[ ! -d "$d" ]]; then
        echo "Error: '$d' is not a directory" >&2
        exit 1
    fi
done

# Temp files for sorted file lists
TMP1=$(mktemp)
TMP2=$(mktemp)

cleanup() {
    rm -f "$TMP1" "$TMP2"
}
trap cleanup EXIT

# Get relative file paths (like ./subdir/file.pt), sorted
(
    cd "$DIR1"
    find . -type f | sort
) > "$TMP1"

(
    cd "$DIR2"
    find . -type f | sort
) > "$TMP2"

echo "=== Files present in '$DIR1' but MISSING in '$DIR2' ==="
comm -23 "$TMP1" "$TMP2" || true
echo

echo "=== Files present in '$DIR2' but MISSING in '$DIR1' ==="
comm -13 "$TMP1" "$TMP2" || true
