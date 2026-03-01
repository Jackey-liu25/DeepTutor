#!/usr/bin/env python3
"""
Request Generation Pipeline

Generate evaluation requests from existing benchmark task entries:
  - one solve request per task
  - one question request per task (exactly one question requested)

Input:
  benchmark/data/generated/benchmark_<timestamp>/*.json

Output:
  benchmark/data/generated/requests_<timestamp>/
    - <entry_id>.json
    - _all_requests.jsonl
    - _summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("benchmark.request_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "benchmark" / "config" / "benchmark_config.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def _to_profile_level(profile: dict[str, Any]) -> str:
    profile_id = str(profile.get("profile_id", "")).lower()
    if "beginner" in profile_id:
        return "beginner"
    if "advanced" in profile_id:
        return "advanced"
    return "intermediate"


def _build_solve_query(entry: dict[str, Any]) -> str:
    task = entry.get("task", {}) or {}
    initial_message = _normalize_space(str(task.get("initial_message", "")))
    if initial_message:
        return initial_message

    title = _normalize_space(str(task.get("title", "")))
    desc = _normalize_space(str(task.get("description", "")))
    if title and desc:
        return f"我在学“{title}”。{desc} 你可以一步步讲解，并给一个具体数字例子吗？"
    if title:
        return f"我在学“{title}”，可以一步步讲解并给一个例子吗？"
    return "我对这个任务对应的知识点不太确定，可以帮我一步步讲清楚并给一个例子吗？"


def _build_question_query(entry: dict[str, Any]) -> str:
    task = entry.get("task", {}) or {}
    profile = entry.get("profile", {}) or {}
    title = _normalize_space(str(task.get("title", ""))) or "当前学习主题"
    level = _to_profile_level(profile)

    if level == "beginner":
        diff_hint = "难度尽量基础一点"
    elif level == "advanced":
        diff_hint = "难度可以稍微有挑战一点"
    else:
        diff_hint = "难度中等"

    return (
        f"我刚学完“{title}”，请基于这个主题给我出一道练习题。"
        f"{diff_hint}，先不要给答案，我做完再看讲解。"
    )


def _build_request_record(entry: dict[str, Any]) -> dict[str, Any]:
    entry_id = str(entry.get("entry_id", "unknown"))
    kb_name = str(entry.get("kb_name", ""))
    task = entry.get("task", {}) or {}

    return {
        "entry_id": entry_id,
        "kb_name": kb_name,
        "task_id": str(task.get("task_id", "")),
        "solve_request": {
            "query": _build_solve_query(entry),
        },
        "question_request": {
            "query": _build_question_query(entry),
        },
    }


def _discover_latest_benchmark_dir(base_output_dir: Path) -> Path:
    candidates = [
        p for p in base_output_dir.glob("benchmark_*")
        if p.is_dir() and p.name[10:].replace("_", "").isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No benchmark_* directories found under: {base_output_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_entries(input_dir: Path, limit: int) -> list[dict[str, Any]]:
    files = sorted(
        p for p in input_dir.glob("*.json")
        if not p.name.startswith("_")
    )
    if limit > 0:
        files = files[:limit]
    if not files:
        raise ValueError(f"No entry JSON files found in: {input_dir}")

    entries: list[dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            entries.append(data)
        else:
            logger.warning("Skip non-dict entry file: %s", path)
    return entries


def _save_requests(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save one request file per entry
    for rec in records:
        entry_id = str(rec.get("entry_id", "unknown"))
        out_path = output_dir / f"{entry_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    # Save combined JSONL
    jsonl_path = output_dir / "_all_requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _save_summary(
    output_dir: Path,
    *,
    input_dir: Path,
    timestamp: str,
    num_entries: int,
) -> Path:
    summary = {
        "timestamp": timestamp,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_entries": num_entries,
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate solve/question requests from benchmark task entries."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Benchmark config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Input benchmark entry directory (e.g. benchmark/data/generated/benchmark_YYYYMMDD_HHMMSS). "
            "If omitted, auto-pick latest benchmark_* directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. If omitted, writes to "
            "benchmark/data/generated/requests_<timestamp>."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N entries (0 = all).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = _load_config(cfg_path)

    base_output_dir = Path(cfg.get("output", {}).get("output_dir", "./benchmark/data/generated"))
    if not base_output_dir.is_absolute():
        base_output_dir = (PROJECT_ROOT / base_output_dir).resolve()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = (PROJECT_ROOT / input_dir).resolve()
    else:
        input_dir = _discover_latest_benchmark_dir(base_output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = base_output_dir / f"requests_{timestamp}"

    entries = _load_entries(input_dir, args.limit)
    logger.info("Loaded %d benchmark entries from %s", len(entries), input_dir)

    records = [_build_request_record(entry) for entry in entries]
    _save_requests(output_dir, records)
    summary_path = _save_summary(
        output_dir,
        input_dir=input_dir,
        timestamp=timestamp,
        num_entries=len(records),
    )

    print("\nDone.")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Summary   : {summary_path}")
    print(f"Requests  : {len(records)}")


if __name__ == "__main__":
    main()
