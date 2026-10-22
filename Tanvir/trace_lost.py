"""Trace "lost_brick" terminations (WRIGGLE_FIT_RL.md §6.6).

Runs a controller on the finger-blocked contexts where releases happen (repo pyramid steps,
family CORNER / POCKET / ENCLOSED) and, for every episode that ends in lost_brick, records per
policy step after the release: the hand's and the brick's lateral error from the true seat,
the brick's height above its seat and its tilt, the scripted post-release phase, and what the
brick touches. Prints one summary line per lost episode and writes results/trace_lost.json.

    python trace_lost.py --controller scripted
    python trace_lost.py --controller results/runs/safe_short/ckpt_0300160.pt
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco                                              # noqa: E402
import numpy as np                                         # noqa: E402

import socket_env as SE                                    # noqa: E402
import socket_family as SF                                 # noqa: E402

OUT = HERE / "results"


def brick_state(env):
    tx, _ = env.true_tgt
    xb = env.d.xpos[env.bid]
    Rb = env.d.xmat[env.bid].reshape(3, 3)
    x, _ = env.hand_pose()
    return {"hand_err_mm": 1e3 * float(np.linalg.norm((x - tx)[:2])),
            "brick_err_mm": 1e3 * float(np.linalg.norm((xb - tx)[:2])),
            "brick_dz_mm": 1e3 * float(xb[2] - (tx[2] - SE.IE.TCP_ABOVE_BOTTOM)),
            "brick_tilt_deg": math.degrees(math.acos(min(1.0, abs(Rb[2, 2]))))}


def touching(env):
    names = set()
    for c in env.d.contact[:env.d.ncon]:
        b1, b2 = env.m.geom_bodyid[c.geom1], env.m.geom_bodyid[c.geom2]
        if env.bid in (b1, b2):
            other = b2 if b1 == env.bid else b1
            names.add(mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_BODY, other) or str(other))
    return sorted(names)


def trace(env, policy, **kw):
    obs = env.reset(**kw)
    done, rows, rel = False, [], None
    while not done:
        a = np.zeros(SE.ACT_DIM) if policy is None else policy(obs["actor"])
        obs, _, done, info = env.step(a)
        if env.released:
            if rel is None:
                rel = env.steps
            post = getattr(env.base, "post", None) if hasattr(env, "base") else None
            rows.append(dict(brick_state(env), step=env.steps, post=post, tips=bool(env.tips_closed),
                             touch=touching(env)))
    return info, rel, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default="scripted")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    if args.controller == "scripted":
        policy, mode, kw = None, "scripted", {}
    else:
        from train_wriggle import Policy
        policy = Policy(args.controller)
        mode = policy.cfg["mode"]
        kw = dict(res_scale=policy.cfg.get("res_scale", 1.0), res_release=policy.cfg.get("res_release", True))
    ctxs = [c for c in SF.repo_contexts() if c["structure"] == "pyramid"]
    pool = SF.pool("train", args.per_class)
    ctxs += [c for cls in ("CORNER", "POCKET", "ENCLOSED") for c in pool[cls]]
    reasons, lost = Counter(), []
    for ctx in ctxs:
        for level in args.levels:
            for s in range(args.seeds):
                env = SE.SocketInsertionEnv(mode=mode, sensing="cheap", seed=500 + s, **kw)
                info, rel, rows = trace(env, policy, context=ctx, level=level, brace="none")
                if rel is None:
                    continue
                reasons[info["reason"]] += 1
                if info["reason"] != "lost_brick":
                    continue
                first, last = rows[0], rows[-1]
                lost.append({"context": ctx["id"], "class": ctx["class"], "level": level, "seed": s,
                             "release_step": rel, "end_step": env.steps, "at_release": first, "at_end": last,
                             "max_brick_err_mm": max(r["brick_err_mm"] for r in rows),
                             "max_hand_err_mm": max(r["hand_err_mm"] for r in rows)})
                print("%-22s L%d s%d  release@%3d end@%3d | at release: hand %.1f brick %.1f mm, dz %.1f mm | "
                      "end: hand %.1f brick %.1f mm, dz %.1f mm, tilt %.1f deg, post %s, touch %s" % (
                          ctx["id"], level, s, rel, env.steps, first["hand_err_mm"], first["brick_err_mm"],
                          first["brick_dz_mm"], last["hand_err_mm"], last["brick_err_mm"], last["brick_dz_mm"],
                          last["brick_tilt_deg"], last["post"], last["touch"]), flush=True)
    print("released episodes by outcome:", dict(reasons))
    OUT.mkdir(exist_ok=True)
    (OUT / "trace_lost.json").write_text(json.dumps({"controller": args.controller, "outcomes": dict(reasons),
                                                     "lost": lost}, indent=1))


if __name__ == "__main__":
    main()
