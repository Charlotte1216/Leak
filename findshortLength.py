"""
findshortLength.py
Scan Train_network logs and rank parameter sets by average episode length.
Outputs a JSON file in the same directory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DIR = Path(
    "C:/Users/Charlotte/NewStart/EXP/GaosiLeak/marl_leakage_search/experiments/Train_network"
)

LOG_PATTERNS = [
    re.compile(
        r"seed(?P<seed>\d+)_agent(?P<agent>[a-z0-9]+)_net(?P<net>[a-z0-9]+)"
        r"_marl(?P<marl>[a-z0-9]+)_lr(?P<lr>[0-9.]+)_gamma(?P<gamma>[0-9.]+)_bs(?P<bs>\d+)\.log$"
    ),
    re.compile(
        r"seed(?P<seed>\d+)_agent(?P<agent>[a-z0-9]+)_marl(?P<marl>[a-z0-9]+)"
        r"_lr(?P<lr>[0-9.]+)_gamma(?P<gamma>[0-9.]+)_bs(?P<bs>\d+)\.log$"
    ),
    re.compile(
        r"seed(?P<seed>\d+)_lr(?P<lr>[0-9.]+)_gamma(?P<gamma>[0-9.]+)_bs(?P<bs>\d+)\.log$"
    ),
]

AVG_LENGTH_RE = re.compile(r"Avg Length:\s*([0-9.]+)")


def _parse_name(name: str) -> Optional[Dict[str, object]]:
    for pattern in LOG_PATTERNS:
        match = pattern.search(name)
        if match:
            info = match.groupdict()
            return {
                "seed": int(info.get("seed", 0)),
                "agent_algorithm": info.get("agent", "unknown"),
                "network_type": info.get("net", "unknown"),
                "marl_algorithm": info.get("marl", "unknown"),
                "lr": float(info.get("lr", 0.0)),
                "gamma": float(info.get("gamma", 0.0)),
                "batch_size": int(info.get("bs", 0)),
            }
    return None


def _extract_avg_lengths(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = AVG_LENGTH_RE.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    continue
    return values


def _rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = sorted(
        results,
        key=lambda r: (r["avg_length"], -r["samples"]),
    )
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx
    return ranked


def _scan_logs(log_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    results: List[Dict[str, object]] = []
    skipped: List[str] = []
    for log_path in log_dir.glob("*.log"):
        meta = _parse_name(log_path.name)
        if meta is None:
            skipped.append(log_path.name)
            continue

        lengths = _extract_avg_lengths(log_path)
        if not lengths:
            skipped.append(log_path.name)
            continue

        avg_length = sum(lengths) / len(lengths)
        record = {
            **meta,
            "avg_length": avg_length,
            "samples": len(lengths),
            "log_file": log_path.name,
        }
        results.append(record)
    return results, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank parameter sets by average episode length.")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_DIR))
    parser.add_argument(
        "--output",
        type=str,
        default="avg_episode_length_ranked.json",
        help="Output JSON filename (written inside log-dir unless absolute).",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log-dir not found: {log_dir}")

    results, skipped = _scan_logs(log_dir)
    ranked = _rank_results(results)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = log_dir / output_path

    payload = {
        "log_dir": str(log_dir),
        "total_logs": len(list(log_dir.glob('*.log'))),
        "ranked": ranked,
        "skipped": skipped,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {output_path}")
    print(f"Ranked: {len(ranked)} | Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
