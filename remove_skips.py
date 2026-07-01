import os
import glob
import re

for filepath in glob.glob('tests/**/*.py', recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if '@pytest.mark.skip(reason="Outdated")' in line:
            continue
        new_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

print("Skips removed.")
