import os
import re
from pathlib import Path

def fix_file(file_path: Path) -> None:
    """Fix line endings and escape sequences in Python file."""
    try:
        # Read with universal newlines mode
        with open(file_path, 'r', encoding='utf-8', newline=None) as f:
            content = f.read()
        
        # Create backup
        with open(str(file_path) + '.bak', 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        # Fix escape sequences
        patterns = [
            (r'(?<!r)(?<!\\)\\S', r'\\S'),
            (r'(?<!r)(?<!\\)\\[', r'\\['),
            (r'(?<!r)(?<!\\)\\]', r'\\]'),
            (r'(?<!r)(?<!\\)\\d', r'\\d'),
            (r'(?<!r)(?<!\\)\\n', r'\\n'),
            (r'(?<!r)(?<!\\)\\t', r'\\t'),
            (r'(?<!r)(?<!\\)\\Z', r'\\Z')
        ]
        
        fixed_content = content
        for pattern, replacement in patterns:
            fixed_content = re.sub(pattern, replacement, fixed_content)
        
        # Write fixed content with LF endings
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(fixed_content)
            
        print(f"Fixed: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main() -> None:
    root_dir = Path(__file__).parent.parent
    for py_file in root_dir.rglob("*.py"):
        if not any(x in str(py_file) for x in ['.venv', '.git', '__pycache__']):
            fix_file(py_file)

if __name__ == "__main__":
    main()