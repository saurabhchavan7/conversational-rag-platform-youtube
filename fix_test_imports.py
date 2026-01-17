from pathlib import Path

test_dir = Path("tests")
header = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""

for test_file in test_dir.glob("test_*.py"):
    content = test_file.read_text(encoding='utf-8')
    
    # Check if already has the fix
    if "sys.path.insert" in content:
        print(f"Skipping {test_file.name} (already has path fix)")
        continue
    
    # Add header
    new_content = header + content
    test_file.write_text(new_content, encoding='utf-8')
    print(f"Fixed {test_file.name}")

print("\nAll test files updated!")