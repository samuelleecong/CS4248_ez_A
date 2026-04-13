"""Render a .mmd file to PNG using mmdc (mermaid-cli via npx)."""
import subprocess, sys
from pathlib import Path

mmd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kd_poster.mmd")
out_path = mmd_path.with_suffix(".png")

result = subprocess.run(
    f'npx -y @mermaid-js/mermaid-cli -i "{mmd_path}" -o "{out_path}" -b white -s 3',
    capture_output=True, text=True, timeout=60, shell=True
)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)
print(f"Rendered to {out_path}")
