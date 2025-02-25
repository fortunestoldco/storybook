#!/bin/bash

# Usage: ./script_name.sh <GitHub repository URL>

# Check if the URL is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <GitHub repository URL>"
    exit 1
fi

REPO_URL="$1"

# Extract the repository name from the URL
REPO_NAME=$(basename -s .git "$REPO_URL")
# Handle URLs that don't end with .git
if [ "$REPO_NAME" = "$REPO_URL" ]; then
    REPO_NAME=$(basename "$REPO_URL")
fi

# Clone the repository
git clone "$REPO_URL"

# Change to the repository directory
cd "$REPO_NAME" || exit

# Output the directory tree
echo "Directory Tree:"

if command -v tree >/dev/null 2>&1; then
    tree .
else
    echo "'tree' command not found. Using 'find' as an alternative."
    find .
fi

# For each file, check if it is a human-readable text file
echo -e "\nContents of human-readable files:\n"

find . -type f | while read -r file; do
    # Check if the file is a text file
    if file "$file" | grep -q ': .*text'; then
        echo "File: $file"
        echo "----------------------------------------"
        cat "$file"
        echo -e "\n"
    fi
done
