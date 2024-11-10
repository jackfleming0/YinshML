# streamlit_run.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from experiments.visualizer import ExperimentDashboard

if __name__ == "__main__":
    dashboard = ExperimentDashboard()
    dashboard.run()

