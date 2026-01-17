"""
Pytest configuration
Ensures imports work correctly from tests/ directory
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"Added to path: {project_root}")