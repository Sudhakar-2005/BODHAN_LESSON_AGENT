"""Phase 2C: generate the machine-readable final report.json.

Aggregates existing Phase 1, 2A, 2B and 2C artifacts only. No API calls, no
recalculation of Phase 2B CER, no mutation of prior artifacts.

Usage::

    uv run python scripts/phase2c_report.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bodhan.reporting import ReportBuilder, write_report

WORKSPACE = Path("data/work/phase1-verify")
EXPORT = Path("data/exports/phase1-verify")
PHASE2A = Path("data/work/phase2a-verify")
ASSEMBLY = Path("data/work/phase2c-verify/audio")
OUTPUT = Path("data/work/phase2c-verify/report.json")


def _commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    value = process.stdout.strip() if process.returncode == 0 else ""
    return value or None


def _working_tree_clean() -> bool | None:
    """Return actual git cleanliness so the commit field is not misleading."""
    try:
        process = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if process.returncode != 0:
        return None
    return not bool(process.stdout.strip())


def main() -> int:
    builder = ReportBuilder(
        work_dir=WORKSPACE,
        export_dir=EXPORT,
        transcribe_dir=PHASE2A / "transcribe",
        evaluation_dir=PHASE2A / "evaluation/cer",
        assembly_dir=ASSEMBLY,
        commit=_commit(),
        working_tree_clean=_working_tree_clean(),
    )
    report = builder.build()
    write_report(OUTPUT, report)

    metadata = report["run_metadata"]
    print(
        f"report: run_id={metadata['run_id']} status={metadata['pipeline_status']} "
        f"target={metadata['target_language']} pages={metadata['page_count']}"
    )
    print(
        f"assembly: {report['audio_assembly']['assembly_status']} "
        f"duration={report['audio_assembly']['total_duration_seconds']}s"
    )
    print(
        "CER (reused): macro="
        f"{report['cer_evaluation']['macro_average_cer_percentage']}% "
        f"micro={report['cer_evaluation']['micro_average_cer_percentage']}%"
    )
    print(
        f"unavailable fields: {len(report['unavailable_fields'])} "
        f"(all reported as null with explicit reasons)"
    )
    print(f"written: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
