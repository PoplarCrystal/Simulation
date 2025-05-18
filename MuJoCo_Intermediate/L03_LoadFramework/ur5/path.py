import sys
from pathlib import Path

current_dir = Path(__file__).parent 

# base_dir = current_dir.parent / "base"   
# sys.path.append(str(base_dir))  

project_dir = current_dir.parent
sys.path.append(str(project_dir))