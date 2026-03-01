from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "MPLCONFIGDIR": str(ROOT / ".mplcache")})


def main() -> None:
    run(["python3", "src/preprocess.py"])
    run(["python3", "src/eda.py"])
    run(["python3", "src/train.py"])
    run(["python3", "src/generate_report.py"])
    run(["python3", "src/generate_detailed_report.py"])
    run(["python3", "src/build_pdf_report.py"])
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
