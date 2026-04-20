#!/usr/bin/env python3
"""
MemMe Benchmark Base Runner

统一的 benchmark 执行框架。各 benchmark 只需实现：
  - load_samples(args) → list[Sample]
  - ingest_sample(store, sample) → None
  - evaluate_sample(config, store, sample) → dict

用法 (子类):
    from run_benchmark_base import BenchmarkRunner, Sample

    class MyBenchmark(BenchmarkRunner):
        name = "MyBenchmark"
        def load_samples(self, args): ...
        def ingest_sample(self, store, sample): ...
        def evaluate_sample(self, config, store, sample): ...

    if __name__ == "__main__":
        MyBenchmark().run()
"""

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add benchmarks/ to path for base_benchmark import
sys.path.insert(0, str(Path(__file__).parent))
from base_benchmark import (
    BenchmarkAPIConfig, add_api_args, build_config, create_store,
    llm_chat, print_config, validate_config,
)


@dataclass
class Sample:
    """A single benchmark sample."""
    id: str
    data: dict  # raw data from the benchmark dataset
    category: str = ""
    db_name: str = ""  # SQLite DB filename (auto-generated if empty)


@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_id: str
    category: str = ""
    correct: bool = False
    prediction: str = ""
    reference: str = ""
    judge_result: str = ""
    ingest_time: float = 0.0
    search_time: float = 0.0
    total_time: float = 0.0
    extra: dict = field(default_factory=dict)


class BenchmarkRunner(ABC):
    """Base class for all MemMe benchmarks."""

    name: str = "Benchmark"
    description: str = ""

    def run(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(description=f"MemMe {self.name}")
        add_api_args(parser)
        parser.add_argument("--data-path", required=True)
        parser.add_argument("--output-dir", default="results")
        parser.add_argument("--cache-dir", default="cache")
        parser.add_argument("--max-questions", type=int, default=0)
        parser.add_argument("--results-only", type=str, default=None)
        self.add_extra_args(parser)
        args = parser.parse_args()

        config = build_config(args)

        # Results-only mode
        if args.results_only:
            self.print_results_only(args.results_only)
            return

        # Load samples
        samples = self.load_samples(args)
        if args.max_questions > 0:
            samples = samples[:args.max_questions]

        # Print header
        print(f"\n{'=' * 60}")
        print(f"  MemMe {self.name}")
        print(f"{'=' * 60}")
        print_config(config)
        print(f"  Samples: {len(samples)} | Workers: {config.workers}")
        print()

        # Validate APIs
        print("Validating APIs...")
        if not validate_config(config):
            print("API validation failed. Fix configuration and retry.")
            return
        print()

        # Setup output
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"run_{ts}.jsonl")
        print(f"Output: {output_path}\n")

        # Run samples
        results = []
        correct_count = 0

        def process_one(si: int, sample: Sample) -> SampleResult:
            t0 = time.time()
            db_name = sample.db_name or f"s_{sample.id}.db"
            db_path = os.path.join(args.cache_dir, db_name)
            skip_ingest = config.reuse_cache and os.path.exists(db_path)

            store = create_store(config, db_path)

            # Ingest
            t_ingest = time.time()
            if not skip_ingest:
                self.ingest_sample(store, sample)
                # Compact + meditate if engine LLM is configured
                if config.engine_api_key:
                    self.compact_and_meditate(store, sample)
                try:
                    store.rebuild_fts_index()
                except Exception:
                    pass
            ingest_time = time.time() - t_ingest

            # Evaluate
            result = self.evaluate_sample(config, store, sample)
            result.ingest_time = ingest_time
            result.total_time = time.time() - t0
            return result

        if config.workers <= 1:
            for si, sample in enumerate(samples):
                result = process_one(si, sample)
                results.append(result)
                if result.correct:
                    correct_count += 1
                status = "+" if result.correct else "-"
                print(f"  [{si+1}/{len(samples)}] {sample.category:<12} {status} "
                      f"{result.total_time:.0f}s (I={result.ingest_time:.0f} S={result.search_time:.1f})",
                      flush=True)
                with open(output_path, "a") as f:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                if (si + 1) % 20 == 0:
                    print(f"  >>> {correct_count}/{si+1} = {correct_count/(si+1)*100:.1f}%")
        else:
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                futures = {pool.submit(process_one, si, s): (si, s) for si, s in enumerate(samples)}
                for future in as_completed(futures):
                    si, sample = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = SampleResult(sample_id=sample.id, category=sample.category)
                        print(f"  [{si+1}] ERROR: {e}")
                    results.append(result)
                    if result.correct:
                        correct_count += 1
                    status = "+" if result.correct else "-"
                    print(f"  [{si+1}/{len(samples)}] {sample.category:<12} {status} "
                          f"{result.total_time:.0f}s (I={result.ingest_time:.0f} S={result.search_time:.1f})",
                          flush=True)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    done = len(results)
                    if done % 20 == 0:
                        print(f"  >>> {correct_count}/{done} = {correct_count/done*100:.1f}%")

        # Summary
        self.print_summary(results)
        print(f"\nResults: {output_path}")

    def compact_and_meditate(self, store, sample: Sample):
        """Run compact + meditate on all sessions. Override if custom logic needed."""
        session_ids = sample.data.get("_session_ids", [])
        for sid in session_ids:
            try:
                store.compact(sid)
            except Exception:
                pass
        user_id = sample.data.get("_user_id", "user")
        try:
            store.meditate(user_id=user_id, triggered_by="benchmark")
        except Exception:
            pass

    def print_summary(self, results: list[SampleResult]):
        """Print summary grouped by category."""
        from collections import defaultdict
        by_cat = defaultdict(lambda: [0, 0])
        for r in results:
            by_cat[r.category][1] += 1
            if r.correct:
                by_cat[r.category][0] += 1

        total_c = sum(v[0] for v in by_cat.values())
        total_n = sum(v[1] for v in by_cat.values())

        print(f"\n{'=' * 60}")
        print(f"  {self.name}: {total_c}/{total_n} = {total_c/max(total_n,1)*100:.1f}%")
        for cat, (c, n) in sorted(by_cat.items()):
            print(f"  {cat:<20} {c:>4}/{n:<4} = {c/max(n,1)*100:.1f}%")
        print(f"{'=' * 60}")

    def print_results_only(self, path: str):
        """Re-generate summary from existing results file."""
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    results.append(SampleResult(**{k: v for k, v in d.items() if k in SampleResult.__dataclass_fields__}))
        self.print_summary(results)

    def add_extra_args(self, parser: argparse.ArgumentParser):
        """Override to add benchmark-specific CLI args."""
        pass

    @abstractmethod
    def load_samples(self, args) -> list[Sample]:
        """Load benchmark data and return list of Samples."""
        ...

    @abstractmethod
    def ingest_sample(self, store, sample: Sample):
        """Ingest sample data into MemoryStore (append_events)."""
        ...

    @abstractmethod
    def evaluate_sample(self, config: BenchmarkAPIConfig, store, sample: Sample) -> SampleResult:
        """Search + answer + judge for one sample. Return SampleResult."""
        ...
