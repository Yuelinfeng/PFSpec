#!/usr/bin/env python3
"""Run a scheduled repeat through an existing evaluate_candidates snapshot."""

from __future__ import annotations

import argparse
from functools import wraps
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


SOURCE_OFFSETS = (0, 8, 15, 23)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate-source", required=True, type=Path)
    parser.add_argument("--repeat-index", type=int, choices=range(4), required=True)
    parser.add_argument("--source-offset", type=int, choices=SOURCE_OFFSETS, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_known_args(argv)


def _method_order(repeat_index):
    return ["w1", "rescue"] if int(repeat_index) % 2 == 0 else ["rescue", "w1"]


def _read_methods(argv):
    for index, item in enumerate(argv):
        if item == "--methods":
            if index + 1 >= len(argv):
                raise ValueError("--methods needs a value")
            return [part.strip() for part in argv[index + 1].split(",") if part.strip()]
        if item.startswith("--methods="):
            return [part.strip() for part in item.partition("=")[2].split(",") if part.strip()]
    return None


def _validate_forwarded_methods(argv, expected):
    actual = _read_methods(argv)
    if actual != expected:
        raise ValueError(
            "Pass --methods " + ",".join(expected) +
            " for this repeat index; all forwarded arguments are preserved."
        )


def rotate_source_groups(prompts, selected_sources, offset):
    sources = list(selected_sources)
    if not sources:
        return list(prompts), sources
    start = int(offset) % len(sources)
    rotated = sources[start:] + sources[:start]
    groups = {source: [] for source in sources}
    for prompt in prompts:
        source = prompt["source"]
        if source not in groups:
            raise ValueError(f"Prompt source {source!r} is absent from selected_sources")
        groups[source].append(prompt)
    reordered = [prompt for source in rotated for prompt in groups[source]]
    if len(reordered) != len(prompts):
        raise AssertionError("Source rotation changed the prompt count")
    return reordered, rotated


def install_source_rotation(module, offset):
    original = module._prompt_rows

    @wraps(original)
    def wrapped(*args, **kwargs):
        prompts, selected_sources = original(*args, **kwargs)
        return rotate_source_groups(prompts, selected_sources, offset)

    module._prompt_rows = wrapped
    return original


def _resolve_candidate(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.name != "evaluate_candidates.py":
            raise ValueError(f"Candidate source must name evaluate_candidates.py: {path}")
        return path
    if path.is_dir():
        candidates = (
            path / "evaluate_candidates.py",
            path / "patches" / "spec_next_20261022" / "evaluate_candidates.py",
            path / "spec_next_20261022" / "evaluate_candidates.py",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Could not locate evaluate_candidates.py under {path}")


def _load_candidate(path):
    spec = importlib.util.spec_from_file_location("rescue_repeat_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option_value(argv, name):
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 >= len(argv):
                return None
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.partition("=")[2]
    return None


def _write_schedule(out_dir, metadata):
    out_dir = Path(out_dir)
    temporary = out_dir / "repeat_schedule.json.tmp"
    destination = out_dir / "repeat_schedule.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, destination)


def _self_test():
    class FakeRunner:
        @staticmethod
        def _prompt_rows():
            sources = [f"scene-{index:02d}" for index in range(30)]
            prompts = []
            for source_index, source in enumerate(sources):
                for prefix in (64, 256):
                    for seed in (52101, 52102):
                        prompts.append(dict(
                            source=source, prefix=prefix, seed=seed,
                            pair_id=f"{source_index:02d}-{prefix}-{seed}",
                        ))
            return prompts, sources

    original_prompts, original_sources = FakeRunner._prompt_rows()
    install_source_rotation(FakeRunner, 15)
    rotated_prompts, rotated_sources = FakeRunner._prompt_rows()
    assert rotated_sources == original_sources[15:] + original_sources[:15]
    assert len(rotated_prompts) == len(original_prompts) == 120
    assert {row["pair_id"] for row in rotated_prompts} == {
        row["pair_id"] for row in original_prompts
    }
    for source in original_sources:
        old_group = [row for row in original_prompts if row["source"] == source]
        new_group = [row for row in rotated_prompts if row["source"] == source]
        assert new_group == old_group
    parsed, forwarded = _parse_args([
        "--candidate-source", "/tmp/source_run02", "--repeat-index", "1",
        "--source-offset", "8", "--project", "/project", "--methods", "rescue,w1",
        "--budget", "512", "--source-scope", "independent",
    ])
    assert parsed.repeat_index == 1 and parsed.source_offset == 8
    assert forwarded == ["--project", "/project", "--methods", "rescue,w1",
                         "--budget", "512", "--source-scope", "independent"]
    _validate_forwarded_methods(forwarded, _method_order(parsed.repeat_index))
    assert _method_order(0) == ["w1", "rescue"]
    assert _method_order(3) == ["rescue", "w1"]
    return dict(status="passed", prompts=120, sources=30,
                source_offset=15, pair_ids_preserved=True,
                within_source_order_preserved=True,
                forwarded_arguments_preserved=True)


def main(argv=None):
    options, forwarded = _parse_args(argv)
    if options.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False))
        return 0

    expected_methods = _method_order(options.repeat_index)
    _validate_forwarded_methods(forwarded, expected_methods)
    candidate_path = _resolve_candidate(options.candidate_source)
    module = _load_candidate(candidate_path)
    install_source_rotation(module, options.source_offset)

    output_arg = _option_value(forwarded, "--out")
    output_dir = Path(output_arg).expanduser().resolve() if output_arg else None
    output_preexisting = bool(output_dir and output_dir.exists())
    original_argv = sys.argv[:]
    sys.argv = [str(candidate_path), *forwarded]
    started = time.time()
    status = "complete"
    failure = None
    result = None
    try:
        result = module.main()
    except BaseException as error:
        status = "failed"
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = original_argv
        if output_dir and not output_preexisting and output_dir.is_dir():
            metadata = dict(
                status=status,
                process_id=os.getpid(),
                started_at_unix=started,
                finished_at_unix=time.time(),
                command=original_argv,
                forwarded_command=[str(candidate_path), *forwarded],
                candidate_source=str(candidate_path),
                repeat_index=options.repeat_index,
                source_offset=options.source_offset,
                method_order=expected_methods,
                source_rotation="selected_sources cyclic order; prompt groups retain their original rows",
                error=failure,
            )
            _write_schedule(output_dir, metadata)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
