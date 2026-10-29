#!/usr/bin/env python
"""GlioAgent CLI: autonomous glioma MRI workup, then an optional co-pilot chat.
Usage and flags: README.md or --help."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from tools.checkpoints import maybe_checkpoints          # light; torch loads later

SAMPLES_CSV = REPO_DIR / "samples" / "samples.csv"
RUNS_DIR = REPO_DIR / "outputs" / "runs"
SEQUENCES = ("t1", "t1ce", "t2", "flair")
SEG_FILE = "hdglio_seg.nii.gz"
TRACE_FILE = "glioagent_trace.json"
REPORT_FILE = "report.json"
TRACE_MD_FILE = "trace.md"

# package -> what breaks without it. A missing one usually means the wrong Python.
REQUIRED_PACKAGES = {
    "torch": "every model",
    "transformers": "Qwen backbone",
    "nibabel": "NIfTI I/O",
    "SimpleITK": "GlioMT preprocessing",
    "monai": "GlioMT",
    "timm": "GlioMT",
    "nnunet": "HD-GLIO",
    "batchgenerators": "HD-GLIO",
    "ants": "MNI registration",
}
BACKBONE_MIN_FREE_MIB = 60_000    # bf16 27B needs ~55 GB
RULE_WIDTH = 78


@dataclass
class Case:
    patient_id: str
    cohort: str | None
    mri_paths: dict[str, str]
    age: int | None
    sex: str | None
    out_dir: str
    ground_truth: str | None = None    # printed only, never fed to the pipeline

    @property
    def seg_path(self) -> str:
        return os.path.join(self.out_dir, SEG_FILE)

    @property
    def trace_path(self) -> str:
        return os.path.join(self.out_dir, TRACE_FILE)

    def problems(self) -> list[str]:
        # blocking problems, empty when runnable
        problems = [f"{name} not found ({_rel(path)})" for name, path in self.mri_paths.items()
                    if not (path and os.path.isfile(path))]
        if self.age is None:
            problems.append("age missing or outside 1-120")
        if self.sex is None:
            problems.append("sex missing or not male/female")
        return problems


# ---- console ----

def _rel(path) -> str:
    # repo-relative when inside the repo
    try:
        return str(Path(path).resolve().relative_to(REPO_DIR))
    except (ValueError, OSError):
        return str(path)


def _banner(title: str) -> None:
    line = "=" * RULE_WIDTH
    print(f"\n{line}\n {title}\n{line}", flush=True)


def _stage(number: int, total: int, title: str) -> None:
    head = f"===== [{number}/{total}] {title} "
    print(f"\n{head}{'=' * max(4, RULE_WIDTH - len(head))}", flush=True)


def _row(label: str, value: str) -> None:
    print(f"  {label:<14} {value}", flush=True)


def _quiet_third_party_logs() -> None:
    # third-party banners, progress bars and deprecation notices bury the trace
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        return
    if not os.environ.get("TRANSFORMERS_VERBOSITY"):
        transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def _print_input_shapes(case: Case) -> None:
    # header-only read: shape + voxel size per sequence
    import nibabel as nib

    print("  [input]", flush=True)
    for name in SEQUENCES:
        path = case.mri_paths[name]
        try:
            header = nib.load(path).header
            shape = "x".join(str(n) for n in header.get_data_shape()[:3])
            voxel = " x ".join(f"{z:g}" for z in header.get_zooms()[:3])
            print(f"    {name:<5} {shape:<16} {voxel} mm   {_rel(path)}")
        except Exception as error:
            print(f"    {name:<5} unreadable ({type(error).__name__})   {_rel(path)}")
    print(f"    {'age/sex':<5} {case.age} y, {case.sex}", flush=True)


def _print_output_shapes(case: Case, trace: dict) -> None:
    # mask + volumes + written artefacts
    import nibabel as nib
    import numpy as np

    print("\n  [output]", flush=True)
    if os.path.isfile(case.seg_path):
        data = nib.load(case.seg_path).get_fdata().astype(int)
        values, counts = np.unique(data, return_counts=True)
        labels = ", ".join(f"{value}:{count}" for value, count in zip(values.tolist(), counts.tolist()))
        print(f"    seg   {'x'.join(str(n) for n in data.shape):<16} labels(voxels) {labels}")
    segmentation = trace.get("tumor_segment") or {}
    print(f"    volumes  ET {segmentation.get('et_cm3')} / NE {segmentation.get('ne_cm3')} / "
          f"WT {segmentation.get('wt_cm3')} cm3")
    findings = (trace.get("structured_findings") or {}).get("findings") or {}
    print(f"    findings {len(findings)} fields, report {len(trace.get('report') or {})} sections")
    for label, path in (("report", os.path.join(case.out_dir, REPORT_FILE)),
                        ("trace", case.trace_path),
                        ("trace md", os.path.join(case.out_dir, TRACE_MD_FILE))):
        print(f"    {label:<8} {_rel(path)}")


def _print_report(trace: dict) -> None:
    # the report text exactly as it lands in report.json
    report = trace.get("report") or {}
    for title, body in (("agent-generated radiology report", report.get("structured_report_text")),
                        ("knowledge-grounded assessment", report.get("clinical_assessment"))):
        text = (body or "").strip()
        if not text:
            continue
        print(f"\n  [{title}]", flush=True)
        for line in text.splitlines():
            print(f"    {line}")
    print(flush=True)


# ---- environment ----

def _check_python_packages() -> None:
    # a wrong interpreter is the usual cause
    missing = [f"{name} ({needed_for})" for name, needed_for in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(f"{sys.executable} is missing: {', '.join(missing)}.\n"
                 "Use the `glioagent` conda env, or set PYTHON to an interpreter that has "
                 "requirements.txt installed.")


def _free_memory_per_gpu() -> dict[int, int]:
    # free MiB per GPU, numbered as this process sees them
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
        free = {int(index): int(mib) for index, mib in
                (line.split(",") for line in query.strip().splitlines())}
    except Exception:
        return {}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return free
    physical = [int(entry) for entry in visible.split(",") if entry.strip().isdigit()]
    return {logical: free[gpu] for logical, gpu in enumerate(physical) if gpu in free}


def _choose_gpus(backbone_gpu: int | None, seg_gpu: int | None) -> tuple[int, int]:
    # backbone on the freest GPU, segmentation on the next
    free = _free_memory_per_gpu()
    freest = sorted(free, key=free.get, reverse=True) or [0]

    def freest_except(taken: int | None) -> int:
        return next((gpu for gpu in freest if gpu != taken), freest[0])

    backbone_gpu = freest_except(seg_gpu) if backbone_gpu is None else backbone_gpu
    seg_gpu = freest_except(backbone_gpu) if seg_gpu is None else seg_gpu
    if free.get(backbone_gpu, BACKBONE_MIN_FREE_MIB) < BACKBONE_MIN_FREE_MIB:
        print(f"  [warn] GPU {backbone_gpu} has {free[backbone_gpu] / 1024:.0f} GB free, "
              "backbone needs ~55 GB", flush=True)
    return backbone_gpu, seg_gpu


def _load_models(model_id: str | None, backbone_gpu: int, seg_gpu: int, with_tools: bool):
    """Load every weight up front so the case run has no hidden loading pauses."""
    import torch
    from agents.llm_client import get_client

    if torch.cuda.is_available():
        torch.cuda.set_device(seg_gpu)    # plain "cuda" tools (GlioMT) follow HD-GLIO

    started = time.time()
    client = get_client(model=model_id, device=f"cuda:{backbone_gpu}")
    _row("ok", f"{client.model_name:<30} GPU {backbone_gpu}   {time.time() - started:>4.0f} s"
               "   orchestrator + T2-FLAIR VQA")
    if not with_tools:
        return client

    started = time.time()
    from tools.gliomt.gliomt_inference import load_bundle
    load_bundle(device="cuda")
    _row("ok", f"{'GlioMT (IDH / 1p19q / grade)':<30} GPU {seg_gpu}   {time.time() - started:>4.0f} s"
               "   molecular heads")

    started = time.time()
    from tools.tool_tumor_segment import warm_hdglio
    warm_hdglio()
    _row("ok", f"{'HD-GLIO (nnU-Net)':<30} GPU {seg_gpu}   {time.time() - started:>4.0f} s"
               "   segmentation")
    _row("ok", f"{'VASARI + MNI atlas':<30} CPU        -      rule-based, no weights")
    return client


# ---- cases ----

def _parse_age(value) -> int | None:
    try:
        age = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return age if 1 <= age <= 120 else None


def _parse_sex(value) -> str | None:
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(
        str(value or "").strip().lower())


def _read_samples() -> list[dict]:
    # samples.csv rows, repo-relative image paths made absolute
    if not SAMPLES_CSV.is_file():
        sys.exit(f"missing {_rel(SAMPLES_CSV)}")
    with open(SAMPLES_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in SEQUENCES:
            column = f"{name}_path"
            if row.get(column) and not os.path.isabs(row[column]):
                row[column] = str(REPO_DIR / row[column])
    return rows


def _collect_case(args: argparse.Namespace) -> Case:
    if args.patient_id:                                  # your own case
        return Case(patient_id=args.patient_id, cohort=None,
                    mri_paths={name: getattr(args, name) for name in SEQUENCES},
                    age=_parse_age(args.age), sex=_parse_sex(args.sex), out_dir=args.out_dir)

    rows = _read_samples()
    if args.sample:
        by_id = {row["sample_id"]: row for row in rows}
        if args.sample not in by_id:
            sys.exit(f"unknown sample '{args.sample}'. Available: {', '.join(by_id)}")
        row = by_id[args.sample]
    else:
        row = rows[0]                                    # no flags: the first shipped sample
    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / row["sample_id"]
    return Case(patient_id=row["sample_id"], cohort=row.get("cohort") or None,
                mri_paths={name: row.get(f"{name}_path", "") for name in SEQUENCES},
                age=_parse_age(row.get("age")), sex=_parse_sex(row.get("sex")),
                out_dir=str(out_dir), ground_truth=row.get("gt_who_2021") or None)


def _print_samples() -> None:
    rows = _read_samples()
    _banner(f"shipped samples ({len(rows)})   {_rel(SAMPLES_CSV)}")
    for row in rows:
        print(f"  {row['sample_id']:<26} {row['age']:>3} y {row['sex']:<7} "
              f"| ground truth: {row.get('gt_who_2021', '?')}")
    print(flush=True)


# ---- runs ----

def _run_case(case: Case, client, backbone_gpu: int, seg_gpu: int, started_at: float) -> None:
    from agents.orchestrator import run_pipeline

    _row("cohort", case.cohort or "?")
    _row("output", _rel(case.out_dir))
    _row("GPUs", f"backbone {backbone_gpu}, HD-GLIO/GlioMT {seg_gpu}")
    if case.ground_truth:
        _row("ground truth", f"{case.ground_truth}   (printed only)")
    _print_input_shapes(case)
    print(flush=True)

    trace = run_pipeline(case.patient_id, case.mri_paths, case.age, case.sex, case.out_dir,
                         cohort=case.cohort, client=client, hdglio_gpu=str(seg_gpu), verbose=True)
    _print_output_shapes(case, trace)
    _print_report(trace)

    molecular = trace.get("molecular") or {}
    _banner(f"done - {case.patient_id}   ({time.time() - started_at:.0f} s)")
    _row("diagnosis", molecular.get("who_2021", "?"))
    if case.ground_truth:
        _row("ground truth", case.ground_truth)
    _row("confidence", trace.get("confidence_tier", "?"))
    print(flush=True)


def _chat(case_dir: str, client) -> None:
    from agents.copilot import load_case, run_turn

    load_case(case_dir)      # raises if the autonomous trace is missing
    print("\n[co-pilot] chat ready. Type 'exit' to quit.\n", flush=True)
    history: list[dict] = []
    while True:
        try:
            message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in ("exit", "quit", ":q"):
            break
        result = run_turn(case_dir, message, history=history, client=client)
        for tool_call in result["tool_calls"]:
            print(f"  [tool: {tool_call['tool']:<16}] {tool_call['brief']}")
        for note in result["grounding"]:
            print(f"  [grounding] {note}")
        if result["report"]:
            print("\n[redrafted report]\n" + "\n".join(f"    {line}" for line in result["report"].splitlines()))
        print(f"\nGlioAgent> {result['reply']}\n", flush=True)
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": result["reply"]}]


# ---- command line ----

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GlioAgent: HD-GLIO segmentation + GlioMT molecular call + VASARI/atlas "
                    "features -> structured WHO 2021 report. With no flags it runs the first "
                    "shipped sample.")

    pick = parser.add_argument_group("what to run (default: the first shipped sample)").add_mutually_exclusive_group()
    pick.add_argument("--list-samples", action="store_true", help="print the shipped demo cases and exit")
    pick.add_argument("--sample", metavar="ID", help="one case from samples/samples.csv")

    own = parser.add_argument_group("your own case (all of these together)")
    own.add_argument("--patient-id")
    for name in SEQUENCES:
        own.add_argument(f"--{name}", metavar="NIFTI")
    own.add_argument("--age", help="years; a GlioMT input, so no default")
    own.add_argument("--sex", help="male or female; a GlioMT input, so no default")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--out-dir", help="output directory (default outputs/runs/<ID>)")
    runtime.add_argument("--llm-gpu", type=int, metavar="N", help="backbone GPU (default: most free memory)")
    runtime.add_argument("--seg-gpu", type=int, metavar="N", help="HD-GLIO/GlioMT GPU (default: next freest)")
    runtime.add_argument("--model", metavar="HF_ID", help="override the backbone model")
    runtime.add_argument("--with-qwen", action="store_true", help="pre-fetch the backbone into the HF cache")
    runtime.add_argument("--no-copilot", action="store_true", help="stop after the report")
    return parser


def _check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # reject combinations argparse cannot express
    if args.list_samples:
        return
    own_given = [name for name in ("patient_id", *SEQUENCES, "age", "sex")
                 if getattr(args, name) is not None]
    if args.sample and own_given:
        parser.error("--patient-id/--t1/--age/--sex describe your own case; "
                     "--sample reads the case from samples.csv")
    if own_given:
        required = ("patient_id", *SEQUENCES, "age", "sex", "out_dir")
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"your own case is missing {', '.join(missing)}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _check_flags(parser, args)
    if args.list_samples:
        _print_samples()
        return 0

    _check_python_packages()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")    # number GPUs like nvidia-smi
    _quiet_third_party_logs()
    started_at = time.time()

    case = _collect_case(args)
    problems = case.problems()
    if problems:
        sys.exit(f"cannot run {case.patient_id}: {'; '.join(problems)}")
    # the co-pilot reuses a finished run; only a missing trace re-runs the pipeline
    reuse_existing = bool(not args.no_copilot and os.path.isfile(case.trace_path))

    _banner(f"GlioAgent - {'co-pilot on an existing run' if reuse_existing else 'autonomous inference'}"
            f"   {case.patient_id}")

    _stage(1, 3, "checkpoints")
    status = maybe_checkpoints(with_qwen=args.with_qwen)
    if not status.get("hdglio") and not reuse_existing:
        sys.exit("HD-GLIO is unavailable; every case is segmented from scratch")
    if not status.get("gliomt") and not reuse_existing:
        print("\n  [warn] GlioMT weights missing -> molecular output stubbed", flush=True)

    _stage(2, 3, "model weights")
    backbone_gpu, seg_gpu = _choose_gpus(args.llm_gpu, args.seg_gpu)
    client = _load_models(args.model, backbone_gpu, seg_gpu, with_tools=not reuse_existing)

    if reuse_existing:
        _stage(3, 3, f"co-pilot - {case.patient_id}")
        _row("reusing", f"{_rel(case.trace_path)}   (delete it to re-run the pipeline)")
    else:
        _stage(3, 3, f"agentic workup - {case.patient_id}")
        _run_case(case, client, backbone_gpu, seg_gpu, started_at)

    if not args.no_copilot:
        _chat(case.out_dir, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
