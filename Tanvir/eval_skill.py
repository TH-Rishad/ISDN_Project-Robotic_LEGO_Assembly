"""Evaluate / select skill-level wriggle policies (WRIGGLE_FIT_RL.md §6.9). Same protocol and output
format as eval_wriggle.py, so the tables compare directly with results/eval/scripted_cheap_v2.md.

  evaluate a checkpoint (deterministic: theta = tanh(mean)); "skill:" prefix optional:
    python eval_skill.py --controller skill:results/runs/skill_main/best.pt --contexts repo family probe \
        --levels 0 2 4 --braces none wriggle --seeds 2 --tag skill_skill_main
  select the best checkpoint of a run on the family validation split (also scores the scripted
  ScriptedWriggle on the same episodes, same seeds); writes best.pt, select.json, select_rows.jsonl:
    python eval_skill.py --select skill_main --per-class 5
Writes Tanvir/results/eval/<tag>.{jsonl,md}.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                         # noqa: E402

import eval_wriggle as EW                                  # noqa: E402
import skill_env as KE                                     # noqa: E402
import socket_env as SE                                    # noqa: E402

OUT = EW.OUT
RUNS = HERE / "results" / "runs"
_POLICY = {}


def _job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    ctrl, ctx, level, brace, seed, sensing, fingers, pad_shift = job
    if ctrl == "scripted":
        env = SE.SocketInsertionEnv(mode="scripted", sensing=sensing, seed=seed, fingers=fingers, pad_shift=pad_shift)
        info = SE.run_episode(env, None, context=ctx, level=level, brace=EW.BRACE_MODE[brace])
    else:
        if ctrl not in _POLICY:
            if ctrl.startswith("es:"):                       # linear ES policy (es_skill.py), numpy only
                from es_skill import ESPolicy
                _POLICY[ctrl] = ESPolicy.load(ctrl[3:])
            else:
                import torch
                torch.set_num_threads(1)
                from train_skill import SkillPolicy
                _POLICY[ctrl] = SkillPolicy(ctrl)
        env = KE.WriggleSkillEnv(sensing=sensing, seed=seed, fingers=fingers, pad_shift=pad_shift)
        info = KE.run_episode(env, _POLICY[ctrl], context=ctx, level=level, brace=EW.BRACE_MODE[brace])
    info = dict(info, brace_condition=brace, seed=seed, source=ctx.get("structure", "family"), kind=ctx.get("kind"),
                controller=ctrl)
    return info


def evaluate(ctrls, ctxs, levels, braces, seeds, workers, sensing="cheap", fingers=True, progress=False,
             pad_shift=False):
    """Every (controller, context, brace, level, seed) in one pool; returns {controller: [rows]}."""
    jobs = []
    for ctrl in ctrls:
        for ctx in ctxs:
            for brace in braces:
                if brace == "planner" and not ctx.get("planner_brace"):
                    continue
                for level in levels:
                    for s in range(seeds):
                        jobs.append((ctrl, ctx, level, brace, 10_000 + s, sensing, fingers, pad_shift))
    t0 = time.time()
    if progress:
        print("%d episodes on %d workers" % (len(jobs), workers), flush=True)
    rows = {c: [] for c in ctrls}
    done = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for info in pool.imap_unordered(_job, jobs, chunksize=1):
            rows[info["controller"]].append(info)
            done += 1
            if progress and done % 50 == 0:
                print("  %d/%d  %.0fs" % (done, len(jobs), time.time() - t0), flush=True)
    return rows


def select(name, workers=7, per_class=5, levels=(0, 2), braces=("none", "wriggle"), seeds=1):
    import shutil
    out = RUNS / name
    cks = sorted(out.glob("ckpt_*.pt"))
    ctxs = EW.contexts(["val"], per_class)
    paths = ["scripted"] + [str(c.resolve()) for c in cks]
    t0 = time.time()
    rows = evaluate(paths, ctxs, levels, braces, seeds, workers, progress=True)
    res = []
    for ck in cks:
        r = rows[str(ck.resolve())]
        res.append(summ(ck.name, r))
    scripted = summ("scripted", rows["scripted"])
    best = max(res, key=lambda r: (r["seated"], -res.index(r)))     # ties go to the earlier checkpoint
    shutil.copy(out / best["ckpt"], out / "best.pt")
    (out / "select.json").write_text(json.dumps({
        "best": best["ckpt"], "scripted": scripted, "results": res, "levels": list(levels), "braces": list(braces),
        "per_class": per_class, "seeds": seeds, "episodes_each": len(rows["scripted"]),
        "wall_s": round(time.time() - t0)}, indent=1))
    with open(out / "select_rows.jsonl", "w") as f:
        for c, rs in rows.items():
            for r in rs:
                f.write(json.dumps({k: r[k] for k in ("controller", "context", "class", "level", "brace_condition",
                                                      "seed", "reason", "success", "steps", "macro_steps", "peak_force_N",
                                                      "u_struct_peak") if k in r}) + "\n")
    print("scripted: %d/%d" % (scripted["seated"], scripted["episodes"]), flush=True)
    for r in res:
        print("select: %s %d/%d" % (r["ckpt"], r["seated"], r["episodes"]), flush=True)
    print(json.dumps(best), flush=True)
    return best


def summ(ckpt, rows):
    return {"ckpt": ckpt, "seated": sum(r["success"] for r in rows), "episodes": len(rows),
            "joint_break": sum(r["reason"] == "joint_break" for r in rows),
            "timeout": sum(r["reason"] == "timeout" for r in rows),
            "by_level": {str(l): sum(r["success"] for r in rows if r["level"] == l) for l in sorted({r["level"] for r in rows})},
            "by_brace": {b: sum(r["success"] for r in rows if r["brace_condition"] == b)
                         for b in sorted({r["brace_condition"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default=None, help="skill:<checkpoint> (or a checkpoint path, or scripted)")
    ap.add_argument("--select", default=None, metavar="RUN", help="select the best checkpoint of this run")
    ap.add_argument("--contexts", nargs="+", default=["repo"], choices=["repo", "family", "val", "probe"])
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--braces", nargs="+", default=["none", "wriggle"], choices=list(EW.BRACE_MODE))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sensing", choices=["cheap", "ft", "none"], default="cheap")
    ap.add_argument("--no-fingers", action="store_true")
    ap.add_argument("--pad-shift", action="store_true", help="grasp off-centre, away from a flush neighbour (§6.10)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.select:
        select(args.select, args.workers, args.per_class)
        return
    c = args.controller
    if c.startswith("skill:"):
        c = c[len("skill:"):]
    es = c.startswith("es:")
    if es:
        c = c[3:]
    ctl = c if c == "scripted" else ("es:" if es else "") + str(Path(c).resolve())
    tag = args.tag or "skill_%s" % Path(c).parent.name
    rows = evaluate([ctl], EW.contexts(args.contexts, args.per_class), args.levels, args.braces, args.seeds,
                    args.workers, args.sensing, not args.no_fingers, progress=True, pad_shift=args.pad_shift)[ctl]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / (tag + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    args.controller = ("es:" if es else "skill:") + c if c != "scripted" else c
    md = EW.report(rows, tag, args).replace("`Tanvir/eval_wriggle.py`", "`Tanvir/eval_skill.py`")
    mac = [r.get("macro_steps", 0) for r in rows]
    md += "\nMean macro steps per episode: %.1f. Controller: `%s`. Pad shift (§6.10): %s.\n" % (
        np.mean(mac), ctl, args.pad_shift)
    (OUT / (tag + ".md")).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
