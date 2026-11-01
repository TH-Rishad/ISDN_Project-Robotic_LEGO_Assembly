"""CPU acceptance verifier.

Exit 0: all three criteria PASS.
Exit 1: a criterion FAIL.
Exit 2: no FAIL, but a required criterion is UNVERIFIED or the evidence is invalid.
"""

import argparse
import json
import sys
from pathlib import Path

KEAN = Path(__file__).resolve().parents[1]
ROOT = KEAN.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Kean.verify_acceptance import (  # noqa: E402
    assess,
    directory_occupied,
    inventory,
    package_from_plans,
    render_md,
    verify_run,
    write_report,
)


def _fresh(parser, out, replace):
    if directory_occupied(out) and not replace:
        parser.error("refusing to overwrite %s; pass a new --out" % out)
    out.mkdir(parents=True, exist_ok=True)


def plan_check(out):
    from Kean.acceptance import ACCEPTANCE, STRATEGY, generate

    plans = {}
    inputs = []
    for name, kind in ACCEPTANCE:
        plan = generate(name, kind)
        plans[name] = plan
        inputs.append({"structure": name, "kind": kind, "image": None})
        path = out / "plans" / ("%s_%s.json" % (name, STRATEGY))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=1, default=str) + "\n")
    (out / "inputs.json").write_text(json.dumps({
        "image_input": False,
        "strategy": STRATEGY,
        "planner": ["planner.sequence", "planner.build_plan", "bracing.assign"],
        "structures": inputs,
    }, indent=1) + "\n")
    return write_report(assess(package_from_plans(plans), "plan-check"), out)


def write_inventory(out):
    payload = inventory(KEAN / "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence_inventory.json").write_text(json.dumps(payload, indent=1) + "\n")
    lines = ["# Acceptance evidence inventory", "", payload["conclusion"], ""]
    for name, row in payload["runs"].items():
        lines.append("## %s" % name)
        lines.append("")
        if not row.get("present"):
            lines.append("Missing.")
            lines.append("")
            continue
        lines.append("- provenance: %s" % row["provenance"])
        lines.append("- cell rows: %s" % row["cell_rows"])
        lines.append("- episodes: %s" % row["episodes"])
        for shape, plan in (row.get("plans") or {}).items():
            lines.append("- %s supports %s (plan brace_required_count %s, %s bricks)" % (
                shape, plan["supports"], plan["brace_required_count"], plan["bricks"]))
        for run in row.get("runs") or []:
            lines.append("- summary %s %s/%s frames %s done %s" % (
                run.get("shape"), run.get("placed"), run.get("total"), run.get("frames"), run.get("done")))
        lines.append("")
    lines.append("Combined versus separate runs: %s" % json.dumps(payload["combined_matches_separate_runs"]))
    lines.append("")
    lines.append("Plan hash matches the separate run: %s" % json.dumps(payload["plan_hash_matches_separate"]))
    lines.append("")
    (out / "evidence_inventory.md").write_text("\n".join(lines))
    print(payload["conclusion"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-check", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if sum([args.plan_check, args.run is not None, args.inventory]) != 1:
        parser.error("choose one of --plan-check, --run, or --inventory")
    if args.inventory:
        write_inventory(args.out or (KEAN / "results"))
        return
    if args.plan_check:
        if args.out is None:
            parser.error("--plan-check requires --out")
        out = args.out if args.out.is_absolute() else ROOT / args.out
        _fresh(parser, out.resolve(), args.replace)
        code = plan_check(out.resolve())
        print("verify: plan-check exit %s" % code)
        raise SystemExit(code)
    run = args.run if args.run.is_absolute() else ROOT / args.run
    out = args.out or (KEAN / "results" / ("verification_" + run.name))
    if not out.is_absolute():
        out = ROOT / out
    _fresh(parser, out.resolve(), args.replace)
    report = verify_run(run.resolve())
    code = write_report(report, out.resolve())
    print(render_md(report))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
