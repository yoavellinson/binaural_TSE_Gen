#!/bin/bash
base_url="https://sofacoustics.org/data/database"
output_dir="./sofa_downloads"

mkdir -p "$output_dir"

# Get all subdirectories (those ending with /)
for subdir in $(wget -q -O - "$base_url/" | grep -oP '(?<=href=")[^"]+/'); do
    subdir_name=$(echo "$subdir" | sed 's#/##g' | tr ' ' '_')
    mkdir -p "$output_dir/$subdir_name"

    echo "Checking $subdir_name ..."

    # Try to find a HRIR/HRTF file first
    first_file=$(wget -q -O - "$base_url/$subdir" | \
        grep -ioP '(?<=href=")[^"]+\.sofa' | \
        grep -iE 'hrir|hrtf' | head -n 1)

    # Fallback: any .sofa file
    if [ -z "$first_file" ]; then
        first_file=$(wget -q -O - "$base_url/$subdir" | \
            grep -ioP '(?<=href=")[^"]+\.sofa' | head -n 1)
        [ -n "$first_file" ] && echo "  → No HRIR/HRTF found, using first .sofa instead"
    fi

    # Download if found
    if [ -n "$first_file" ]; then
        echo "  → Downloading $first_file"
        wget -q -O "$output_dir/$subdir_name/$first_file" "$base_url/$subdir$first_file"
    else
        echo "  → No .sofa files found at all"
    fi
done
