#!/bin/bash

# Script to run mogrify to fix colour profiles in images and then rename all images in a numbered scheme

folder_name="FOLDER_NAME"
base_path="BASE_PATH"

folder_path="$base_path/$folder_name"
count=1

# Change directory to the target folder
cd "$folder_path"

# Run mogrify
mogrify *

# Rename files
for file in *; do
    if [ -f "$file" ]; then
        extension="${file##*.}"
        new_name="$count.$extension"
        mv "$file" "$new_name"
        ((count++))
    fi
done

