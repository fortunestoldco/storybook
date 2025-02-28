import os
import re
from pathlib import Path

def fix_file(file_path: Path):
    """Fix line endings and escape sequences in Python file."""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        with open(str(file_path) + '.bak', 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Fix escape sequences in non-raw strings
        fixed_content = content
        patterns = [
            (r'(?<!r)(?<!\\)\\S', r'\\S'),
            (r'(?<!r)(?<!\\)\\[', r'\\['),
            (r'(?<!r)(?<!\\)\\]', r'\\]'),
            (r'(?<!r)(?<!\\)\\d', r'\\d'),
            (r'(?<!r)(?<!\\)\\n', r'\\n'),
            (r'(?<!r)(?<!\\)\\t', r'\\t'),
            (r'(?<!r)(?<!\\)\\Z', r'\\Z')
        ]
        
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(fixed_content)
            
        print(f"Fixed: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Get the project root directory
    root_dir = Path(__file__).parent
    
    # Process all Python files except those in .venv
    for py_file in root_dir.rglob("*.py"):
        if ".venv" not in str(py_file) and ".git" not in str(py_file):
            fix_file(py_file)

if __name__ == "__main__":
    main()