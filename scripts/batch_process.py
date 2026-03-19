"""
Vrite - Batch Processor
Generate multiple videos unattended from a JSON or CSV job file.

Usage:
    python scripts/batch_process.py --jobs scripts/sample_jobs.json
    python scripts/batch_process.py --jobs jobs.csv --output-dir outputs/batch/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vrite.config import PipelineConfig
from vrite.utils import setup_logging, format_duration
from run import VideoPipeline

log = logging.getLogger("vrite.batch")


@dataclass
class BatchJob:
    sample_video: str
    script: str
    output: str
    sample_audio: str = None
    job_id: int = 0


@dataclass
class BatchResult:
    job_id: int
    output: str
    success: bool
    error: str = None
    elapsed: float = 0.0


def load_jobs(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Job file not found: {path}")
    if p.suffix.lower() == ".json":
        records = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("JSON job file must be a list.")
    elif p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    jobs = []
    for i, rec in enumerate(records):
        script = rec.get("script", "").strip()
        sf = rec.get("script_file", "").strip()
        if not script and sf:
            sfp = Path(sf)
            if not sfp.exists():
                raise FileNotFoundError(f"Script file not found: {sf}")
            script = sfp.read_text(encoding="utf-8")
        if not script:
            raise ValueError(f"Job {i} has no script.")
        output = (rec.get("output", "").strip()
                  or f"outputs/batch/job_{i:04d}.mp4")
        jobs.append(BatchJob(
            sample_video=rec["sample_video"],
            script=script,
            output=output,
            sample_audio=rec.get("sample_audio") or None,
            job_id=i))
    return jobs


def run_job(job: BatchJob, cfg: PipelineConfig) -> BatchResult:
    t0 = time.perf_counter()
    try:
        result = VideoPipeline(cfg).run(
            sample_video=job.sample_video,
            script=job.script,
            sample_audio=job.sample_audio,
            output_path=job.output)
        return BatchResult(
            job_id=job.job_id, output=result.output_path,
            success=True, elapsed=time.perf_counter() - t0)
    except Exception as exc:
        log.exception("Job %d failed: %s", job.job_id, exc)
        return BatchResult(
            job_id=job.job_id, output=job.output,
            success=False, error=str(exc),
            elapsed=time.perf_counter() - t0)


def run_batch(jobs: list, cfg: PipelineConfig,
              workers: int = 1) -> list:
    results = []
    if workers == 1:
        for i, job in enumerate(jobs):
            log.info("Job %d/%d ...", i + 1, len(jobs))
            r = run_job(job, cfg)
            results.append(r)
            log.info("%s Job %d in %s",
                     "OK" if r.success else "FAIL",
                     r.job_id, format_duration(r.elapsed))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_job, j, cfg): j for j in jobs}
            for fut in as_completed(futures):
                results.append(fut.result())
    return sorted(results, key=lambda r: r.job_id)


def print_summary(results: list) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.success)
    elapsed = sum(r.elapsed for r in results)
    print(f"\n{'='*50}")
    print(f"  {ok}/{total} succeeded  "
          f"({format_duration(elapsed)} total)")
    print(f"{'='*50}")
    for r in results:
        icon = "OK  " if r.success else "FAIL"
        print(f"  [{icon}] Job {r.job_id:3d}  "
              f"{format_duration(r.elapsed):>8s}  {r.output}")
        if r.error:
            print(f"         Error: {r.error}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Vrite batch processor")
    p.add_argument("--jobs", required=True,
                   help="JSON or CSV job file")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--workers", default=1, type=int)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--quality", default=20, type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(debug=args.verbose)

    try:
        jobs = load_jobs(args.jobs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Job file error: {exc}", file=sys.stderr)
        return 1

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            job.output = str(out / Path(job.output).name)

    log.info("Loaded %d job(s)", len(jobs))

    if args.dry_run:
        print(f"Dry run: {len(jobs)} jobs OK.")
        return 0

    cfg = PipelineConfig(device=args.device, output_crf=args.quality)
    results = run_batch(jobs, cfg, workers=args.workers)
    print_summary(results)
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
