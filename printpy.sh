#!/bin/bash

# Script to recursively find and display Python files with filename headers
# Usage: ./show_py_files.sh [starting_directory]

# Set the starting directory (default to current directory if not provided)
start_dir="${1:-.}"

# Find all Python files recursively
find "$start_dir" -type f -name "*.py" | sort | while IFS= read -r file; do
    # Calculate terminal width for the header line
    term_width=$(tput cols 2>/dev/null || echo 80)

    # Create a header with the filename
    filename=$(basename "$file")
    filepath=$(dirname "$file")
    header="=== $filepath/$filename "

    # Fill the rest of the line with = characters
    printf "%s" "$header"
    printf '=%.0s' $(seq $((term_width - ${#header})))
    printf "\n\n"

    # Output the file contents
    cat "$file" || echo "Error: Could not read $file"

    # Add a separator after each file
    printf "\n\n"
done
