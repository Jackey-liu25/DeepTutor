#!/usr/bin/env python3
"""
Initialize one KB per PDF from ../documents, then generate profiles.

Flow per PDF:
  1) Create KB (name derived from PDF filename)
  2) Process document into RAG KB
  3) Generate knowledge scope
  4) Generate student profiles
  5) Save outputs under benchmark/data/generated/profiles_from_documents_<timestamp>/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Ensure project root is importable when script is executed directly.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT_FOR_IMPORT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from benchmark.data_generation.profile_generator import generate_profiles_for_kb
from benchmark.data_generation.scope_generator import generate_knowledge_scope
from src.knowledge.initializer import KnowledgeBaseInitializer

logger = logging.getLogger("benchmark.init_kbs_profiles")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "benchmark" / "config" / "benchmark_config.yaml"
DEFAULT_DOCS_DIR = PROJECT_ROOT.parent / "documents"
MAX_CONCURRENCY = 5


class PipelineAbortError(RuntimeError):
    """Abort all processing when any single PDF pipeline fails."""


def _sanitize_kb_name(name: str) -> str:
    """Convert filename stem into a valid kb_name."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "kb"


def _unique_name(base: str, used: set[str]) -> str:
    """Make KB name unique in current run."""
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cleanup_failed_kb_data(kb_base_dir: Path, kb_name: str, output_dir: Path) -> None:
    """Remove artifacts for a failed PDF pipeline."""
    kb_dir = kb_base_dir / kb_name
    output_json = output_dir / f"{kb_name}.json"

    if output_json.exists():
        try:
            output_json.unlink()
            logger.info("Removed failed output file: %s", output_json)
        except Exception as e:
            logger.warning("Failed to remove output file %s: %s", output_json, e)

    if kb_dir.exists():
        try:
            shutil.rmtree(kb_dir)
            logger.info("Removed failed KB directory: %s", kb_dir)
        except Exception as e:
            logger.warning("Failed to remove KB directory %s: %s", kb_dir, e)


def _parse_gpu_ids(raw: str) -> list[str]:
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    return ids or ["0"]


async def _process_pdf(
    *,
    pdf_path: Path,
    kb_name: str,
    kb_base_dir: Path,
    profile_cfg: dict,
    rag_cfg: dict,
    output_dir: Path,
    skip_extract: bool,
    use_mineru_api: bool,
    mineru_api_token: str | None,
    mineru_model_version: str,
) -> dict:
    """Initialize KB from one PDF and generate profiles."""
    logger.info("=" * 70)
    logger.info("PDF: %s", pdf_path.name)
    logger.info("KB : %s", kb_name)
    logger.info("=" * 70)

    initializer = KnowledgeBaseInitializer(
        kb_name=kb_name,
        base_dir=str(kb_base_dir),
    )
    initializer.create_directory_structure()
    copied = initializer.copy_documents([str(pdf_path)])
    if not copied:
        raise RuntimeError(f"Failed to copy PDF: {pdf_path}")

    await initializer.process_documents(
        use_mineru_api=use_mineru_api,
        mineru_api_token=mineru_api_token,
        mineru_model_version=mineru_model_version,
    )
    if not skip_extract:
        initializer.extract_numbered_items()

    scope = await generate_knowledge_scope(
        kb_name=kb_name,
        seed_queries=rag_cfg.get("seed_queries"),
        mode=rag_cfg.get("mode", "naive"),
        kb_base_dir=str(kb_base_dir),
    )

    profiles = await generate_profiles_for_kb(
        knowledge_scope=scope,
        background_types=profile_cfg.get(
            "background_types", ["beginner", "intermediate", "advanced"]
        ),
        profiles_per_kb=profile_cfg.get("profiles_per_subtopic", 3),
    )

    out = {
        "pdf_file": str(pdf_path),
        "kb_name": kb_name,
        "knowledge_scope": scope,
        "profiles": profiles,
        "num_profiles": len(profiles),
    }
    out_path = output_dir / f"{kb_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)
    return out


async def _run_single_pdf_child(
    *,
    pdf_path: Path,
    kb_name: str,
    kb_base_dir: Path,
    profile_cfg: dict,
    rag_cfg: dict,
    output_dir: Path,
    skip_extract: bool,
    use_mineru_api: bool,
    mineru_api_token: str | None,
    mineru_model_version: str,
) -> None:
    """Run one PDF pipeline in child process; cleanup and fail on error."""
    try:
        await _process_pdf(
            pdf_path=pdf_path,
            kb_name=kb_name,
            kb_base_dir=kb_base_dir,
            profile_cfg=profile_cfg,
            rag_cfg=rag_cfg,
            output_dir=output_dir,
            skip_extract=skip_extract,
            use_mineru_api=use_mineru_api,
            mineru_api_token=mineru_api_token,
            mineru_model_version=mineru_model_version,
        )
    except Exception as e:
        logger.exception("Failed on %s -> %s: %s", pdf_path.name, kb_name, e)
        _cleanup_failed_kb_data(kb_base_dir=kb_base_dir, kb_name=kb_name, output_dir=output_dir)
        raise PipelineAbortError(
            f"Pipeline failed for {pdf_path.name} (kb={kb_name}). Program terminated."
        ) from e


async def _run_jobs_with_gpu_sharding(
    *,
    jobs: list[tuple[Path, str]],
    gpu_ids: list[str],
    config_path: Path,
    output_dir: Path,
    skip_extract: bool,
    use_mineru_api: bool,
    mineru_api_token: str | None,
    mineru_model_version: str,
) -> list[dict]:
    """Run jobs as child processes pinned to GPUs with fail-fast behavior."""
    available_gpus = list(gpu_ids)
    running: list[dict] = []
    waiting_jobs = list(jobs)
    results: list[dict] = []

    async def _start_one(pdf_path: Path, kb_name: str, gpu_id: str) -> dict:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-pdf",
            str(pdf_path),
            "--single-kb-name",
            kb_name,
            "--single-output-dir",
            str(output_dir),
            "--config",
            str(config_path),
            "--mineru-model-version",
            mineru_model_version,
            "--gpu-ids",
            gpu_id,
        ]
        if skip_extract:
            cmd.append("--skip-extract")
        if use_mineru_api:
            cmd.append("--use-mineru-api")
        else:
            cmd.append("--no-use-mineru-api")
        if mineru_api_token:
            cmd.extend(["--mineru-api-token", mineru_api_token])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        logger.info("Spawn %s on GPU %s", kb_name, gpu_id)
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        wait_task = asyncio.create_task(proc.wait())
        return {
            "pdf_path": pdf_path,
            "kb_name": kb_name,
            "gpu_id": gpu_id,
            "proc": proc,
            "wait_task": wait_task,
        }

    while waiting_jobs or running:
        while waiting_jobs and available_gpus:
            pdf_path, kb_name = waiting_jobs.pop(0)
            gpu_id = available_gpus.pop(0)
            running.append(await _start_one(pdf_path, kb_name, gpu_id))

        if not running:
            break

        done, _ = await asyncio.wait(
            [r["wait_task"] for r in running],
            return_when=asyncio.FIRST_COMPLETED,
        )

        finished_entries = [r for r in running if r["wait_task"] in done]
        for entry in finished_entries:
            running.remove(entry)
            available_gpus.append(entry["gpu_id"])

            exit_code = entry["wait_task"].result()
            if exit_code != 0:
                logger.error(
                    "Job failed: %s (pdf=%s) on GPU %s, exit_code=%s",
                    entry["kb_name"],
                    entry["pdf_path"].name,
                    entry["gpu_id"],
                    exit_code,
                )
                for other in running:
                    other["proc"].terminate()
                await asyncio.gather(*[r["wait_task"] for r in running], return_exceptions=True)
                raise PipelineAbortError(
                    f"Pipeline failed for {entry['pdf_path'].name} (kb={entry['kb_name']})."
                )

            output_json = output_dir / f"{entry['kb_name']}.json"
            if output_json.exists():
                with open(output_json, encoding="utf-8") as f:
                    results.append(json.load(f))

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize one KB per PDF in ../documents, then generate profiles."
    )
    parser.add_argument(
        "--docs-dir",
        default=str(DEFAULT_DOCS_DIR),
        help=f"Directory containing PDF files (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Benchmark config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip numbered items extraction for faster initialization",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N PDFs (0 = all)",
    )
    parser.add_argument(
        "--use-mineru-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MinerU cloud API for parsing (default: True). Use --no-use-mineru-api for local parser.",
    )
    parser.add_argument(
        "--mineru-api-token",
        default=None,
        help="MinerU API token (falls back to MINERU_API_TOKEN env var if omitted).",
    )
    parser.add_argument(
        "--mineru-model-version",
        default="vlm",
        help='MinerU API model version (default: "vlm").',
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help='GPU ids for sharding workloads, comma-separated (default: "0,1,2,3").',
    )
    parser.add_argument("--single-pdf", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-kb-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-output-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = _load_config(cfg_path)

    kb_base_dir = Path(
        cfg.get("knowledge_bases", {}).get("base_dir", "./data/knowledge_bases")
    )
    if not kb_base_dir.is_absolute():
        kb_base_dir = (PROJECT_ROOT / kb_base_dir).resolve()

    profile_cfg = cfg.get("profile_generation", {})
    rag_cfg = cfg.get("rag_query", {})

    if args.single_pdf:
        if not args.single_kb_name or not args.single_output_dir:
            raise ValueError("--single-pdf mode requires --single-kb-name and --single-output-dir")
        pdf_path = Path(args.single_pdf).resolve()
        output_dir = Path(args.single_output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        await _run_single_pdf_child(
            pdf_path=pdf_path,
            kb_name=args.single_kb_name,
            kb_base_dir=kb_base_dir,
            profile_cfg=profile_cfg,
            rag_cfg=rag_cfg,
            output_dir=output_dir,
            skip_extract=args.skip_extract,
            use_mineru_api=args.use_mineru_api,
            mineru_api_token=args.mineru_api_token,
            mineru_model_version=args.mineru_model_version,
        )
        return

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_absolute():
        docs_dir = (PROJECT_ROOT / docs_dir).resolve()
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents dir not found: {docs_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "benchmark" / "data" / "generated" / f"profiles_from_documents_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise ValueError(f"No PDF files found in: {docs_dir}")

    used_names: set[str] = set()
    jobs = []
    for pdf in pdfs:
        kb_name = _unique_name(_sanitize_kb_name(pdf.stem), used_names)
        jobs.append((pdf, kb_name))

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    logger.info("Starting pipelines with max concurrency = %d", min(MAX_CONCURRENCY, len(gpu_ids)))
    logger.info("GPU sharding enabled on ids: %s", ",".join(gpu_ids))
    logger.info(
        "Parser mode: %s",
        "MinerU cloud API" if args.use_mineru_api else "local parser",
    )
    try:
        results = await _run_jobs_with_gpu_sharding(
            jobs=jobs,
            gpu_ids=gpu_ids[:MAX_CONCURRENCY],
            config_path=cfg_path,
            output_dir=output_dir,
            skip_extract=args.skip_extract,
            use_mineru_api=args.use_mineru_api,
            mineru_api_token=args.mineru_api_token,
            mineru_model_version=args.mineru_model_version,
        )
    except PipelineAbortError as e:
        logger.error("Aborting due to pipeline failure: %s", e)
        raise SystemExit(1)

    summary = {
        "timestamp": timestamp,
        "docs_dir": str(docs_dir),
        "kb_base_dir": str(kb_base_dir),
        "num_pdfs": len(pdfs),
        "num_success": len(results),
        "results": [
            {
                "pdf_file": r["pdf_file"],
                "kb_name": r["kb_name"],
                "num_profiles": r["num_profiles"],
            }
            for r in results
        ],
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Success: {len(results)}/{len(pdfs)} PDFs")


if __name__ == "__main__":
    asyncio.run(main())
