from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    feasible: bool
    evidence: list[str]
    missing: list[str]
    recommendation: str


def csv_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        names = set(reader.fieldnames or [])
    return columns.issubset(names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate 3-way comparison feasibility")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("three_way_comparison_audit.json"))
    args = parser.parse_args()

    repo = args.repo

    evidence_files = {
        "teacher_fixed": repo / "fixed_results.csv",
        "teacher_robust": repo / "fft_results_robust.csv",
        "student_val": repo / "runs" / "attention_res1d" / "best_val_predictions.csv",
        "student_full": repo / "runs" / "final_model" / "ultimate_final_result.csv",
        "grouped_compare": repo / "grouped_val_comparison.csv",
        "distill_npz": repo / "dataset_distill.npz",
        "distill_meta": repo / "dataset_distill_meta.csv",
        "raw_baseline_candidate": repo / "clean_data.csv",
        "phase_baseline_candidate": repo / "phase_check_results.csv",
    }

    checks: list[CheckResult] = []

    # 1) student vs teacher
    s_t_evidence: list[str] = []
    s_t_missing: list[str] = []
    grouped_ok = csv_has_columns(
        evidence_files["grouped_compare"],
        {"Filename", "FFT_Width", "Student_Width", "Teacher_Label_Width"},
    )
    if grouped_ok:
        s_t_evidence.append("grouped_val_comparison.csv has teacher+student columns")
    else:
        s_t_missing.append("grouped_val_comparison.csv with teacher/student fields")

    val_ok = csv_has_columns(
        evidence_files["student_val"],
        {"Filename", "Pred_Width", "True_Width"},
    )
    if val_ok:
        s_t_evidence.append("runs/attention_res1d/best_val_predictions.csv has Pred_Width and True_Width")
    else:
        s_t_missing.append("runs/attention_res1d/best_val_predictions.csv with Pred_Width/True_Width")

    checks.append(
        CheckResult(
            name="Student vs Teacher",
            feasible=(grouped_ok or val_ok),
            evidence=s_t_evidence,
            missing=s_t_missing,
            recommendation="Can be used in formal experiments if split leakage is controlled and teacher source is clearly declared.",
        )
    )

    # 2) student vs non-teacher baseline
    baseline_ok = False
    b_evidence: list[str] = []
    b_missing: list[str] = []
    # strict rule: independent baseline must explicitly provide ground-truth columns/provenance
    strict_gt_cols = {"Filename", "GroundTruth_Width"}
    for key in ["raw_baseline_candidate", "phase_baseline_candidate"]:
        path = evidence_files[key]
        if csv_has_columns(path, strict_gt_cols):
            baseline_ok = True
            b_evidence.append(f"{path.name} has explicit ground-truth width columns")
        elif path.exists():
            b_evidence.append(f"{path.name} exists but lacks explicit ground-truth provenance columns")

    if not baseline_ok:
        b_missing.append("No file with explicit non-teacher ground-truth target linked to student predictions")

    checks.append(
        CheckResult(
            name="Student vs Non-teacher baseline",
            feasible=baseline_ok,
            evidence=b_evidence,
            missing=b_missing,
            recommendation=(
                "Not rigorous unless baseline labels come from an independently measured target "
                "(e.g., instrument/encoder truth) and are connected by Filename."
            ),
        )
    )

    # 3) teacher vs independent ground truth
    gt_ok = False
    gt_evidence: list[str] = []
    gt_missing: list[str] = []

    # heuristic: independent GT should have filename + measured position/width columns, not teacher-only names
    gt_candidates = [
        repo / "clean_data.csv",
        repo / "linear_fit_data.csv",
        repo / "phase_check_results.csv",
    ]
    for p in gt_candidates:
        if not p.exists():
            continue
        if csv_has_columns(p, {"Filename", "Position_mm"}):
            gt_evidence.append(f"{p.name} has Filename+Position_mm")
            # still may not be independent; cannot verify provenance from csv alone

    if len(gt_evidence) == 0:
        gt_missing.append("No explicit independent ground-truth table with provenance")

    checks.append(
        CheckResult(
            name="Teacher vs Independent ground truth",
            feasible=gt_ok,
            evidence=gt_evidence,
            missing=gt_missing,
            recommendation="Do not claim true-ground-truth accuracy without external measurement provenance.",
        )
    )

    out = {
        "repo": str(repo.resolve()),
        "files_checked": {k: str(v) for k, v in evidence_files.items()},
        "checks": [
            {
                "name": c.name,
                "feasible": c.feasible,
                "evidence": c.evidence,
                "missing": c.missing,
                "recommendation": c.recommendation,
            }
            for c in checks
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Three-way comparison feasibility audit ===")
    for c in checks:
        print(f"- {c.name}: {'YES' if c.feasible else 'NO'}")
        for ev in c.evidence:
            print(f"    evidence: {ev}")
        for ms in c.missing:
            print(f"    missing : {ms}")
    print(f"saved: {args.out}")

    # overall non-zero only if core comparison #1 cannot run
    return 0 if checks[0].feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())