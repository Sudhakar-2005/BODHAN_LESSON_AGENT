"""Print the run manifest as a compact stage table (reporting helper)."""

from __future__ import annotations

import json
from pathlib import Path

manifest = json.loads(Path("data/work/sample-chapter/manifest.json").read_text(encoding="utf-8"))
print("status:", manifest["status"])
for name, entry in manifest["stages"].items():
    artifacts = json.dumps(entry.get("artifacts", {}), ensure_ascii=True)
    print(f"  {name:11s} {entry['status']:6s} {entry['duration_s']:7.2f}s {artifacts[:160]}")
