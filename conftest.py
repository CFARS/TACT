"""
Pytest configuration file to ensure the tact package can be imported during tests.
"""
import sys
from pathlib import Path

# Add the project root to Python path so 'tact' can be imported
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
