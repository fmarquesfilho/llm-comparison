import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Configura ambiente do projeto"""
    try:
        subprocess.run(['conda', 'env', 'create', '-f', '../environment.yml'], check=True)
        print("Environment created successfully.")
    except subprocess.CalledProcessError:
        print("Error creating conda environment.")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()