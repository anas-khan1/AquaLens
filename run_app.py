"""Convenience launcher for the AquaLens Streamlit App."""
import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent
    app_file = repo_root / "app" / "streamlit_app.py"
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
    ]
    print(f"Starting AquaLens Marine AI App from {repo_root}...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root))

if __name__ == "__main__":
    main()

