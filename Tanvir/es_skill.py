"""Paired evolution-strategy search for the skill-level wriggle policy (WRIGGLE_FIT_RL.md §6.11).

PPO on macro steps (§6.9) found no improvement over theta = 0, so this searches a small policy directly:

    theta = clip(b[socket class] + W @ z, -1, 1)      b: 6 x 10 (per-class bias), W: 10 x 8, 140 parameters, init 0

z = 8 features of the macro observation the actor already sees (no privileged information): depth progress,
its change over the window, mean cheap-wrench z, mean lateral cheap-wrench magnitude, impedance tracking-error
magnitude, elapsed time, released flag, brace active. Zero parameters give theta = 0 exactly (the scripted
controller).

Search: OpenAI-ES with mirrored sampling (+-sigma*eps), centred-rank fitness, Adam on the mean. Common random
numbers: every candidate of a generation is run on the SAME batch of episodes (same contexts, aim-error levels,
braces and env seeds; a fresh batch per generation from the TRAIN split, one episode per socket class x level
{0,1,2,3}, brace none / ideal alternating over the grid). The mean and (every few generations) theta = 0 are scored on the
same batch, so improvement is measured as a paired difference. Fitness = mean episode return (the env's
sparse +-1 reward plus the neighbour-utilisation penalty and the 0.002 per tick time cost; no shaping).

    python es_skill.py --name es_main --max-hours 1.5            # search
    python es_skill.py --name es_main --select                   # pick the best of the last means on validation
Writes results/runs/<name>/{train.jsonl, mean_genNNN.json, zero.json, avg_last.json, select.json, best.json}.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                         # noqa: E402

import skill_env as KE                                     # noqa: E402
import socket_family as SF                                 # noqa: E402

RUNS = HERE / "results" / "runs"
NC, NT, NF = len(SF.CLASSES), KE.THETA_DIM, 8
N_B, N_W = NC * NT, NT * NF
N_PARAMS = N_B + N_W
LEVELS = (0, 1, 2, 3)
CTX_END = KE.SE.BRACE_DIM                  # the actor obs ends with ctx (32) + brace (9)
CLASS_SLICE = slice(-(KE.SE.CTX_DIM + KE.SE.BRACE_DIM) + 8, -(KE.SE.CTX_DIM + KE.SE.BRACE_DIM) + 14)


def features(o):
    """8 normalised features of the macro observation (layout: skill_env._macro_obs)."""
    z = np.array([o[0],                                     # depth progress (10 mm units)
                  2.0 * o[1],                               # its change over the window
                  3.0 * o[4],                               # mean cheap wrench, hand-frame z (/30 N)
                  3.0 * np.hypot(o[2], o[3]),               # mean lateral wrench magnitude
                  0.5 * np.linalg.norm(o[20:23]),           # impedance tracking-error magnitude
                  o[43],                                    # elapsed time (fraction of the 10 s episode)
                  o[39],                                    # released flag
                  o[-CTX_END]])                             # brace active
    return np.clip(z, -3.0, 3.0)


class ESPolicy:
    def __init__(self, params):
        self.p = np.asarray(params, float)
        self.b = self.p[:N_B].reshape(NC, NT)
        self.W = self.p[N_B:].reshape(NT, NF)

    def __call__(self, o):
        c = int(np.argmax(o[CLASS_SLICE]))
        return np.clip(self.b[c] + self.W @ features(o), -1.0, 1.0)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["params"])


def save_params(path, p, **extra):
    Path(path).write_text(json.dumps(dict(params=[float(x) for x in p], **extra)))


# --- the common-random-number batch ----------------------------------------------------------

def make_batch(pool, gen, seed=0):
    """24 episodes: one per (socket class, aim-error level); the context is a random train socket of the class,
    the brace alternates none / ideal over the grid; the env seed is common to all candidates."""
    rng = np.random.default_rng([seed, gen])
    batch = []
    for ci, cls in enumerate(SF.CLASSES):
        for li, level in enumerate(LEVELS):
            ctx = pool[cls][rng.integers(len(pool[cls]))]
            brace = "none" if (ci + li) % 2 == 0 else "ideal"
            batch.append((ctx, level, brace, int(rng.integers(1 << 30))))
    return batch


def _job(job):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    cand, params, ctx, level, brace, seed, pad_shift = job
    env = KE.WriggleSkillEnv(seed=seed, shaping=0.0, pad_shift=pad_shift)
    pol = ESPolicy(params)
    obs = env.reset(context=ctx, level=level, brace=brace)
    ret, done = 0.0, env.dead is not None
    if done:
        ret = env.dead[0]
        info = env.dead[1]
    while not done:
        obs, r, done, info = env.step(pol(obs["actor"]))
        ret += r
    return cand, ret, bool(info["success"]), info["reason"], info["class"]


def build_jobs(cands, batch, pad_shift):
    """Common random numbers: each candidate gets exactly the batch's (context, level, brace, env seed) tuples."""
    return [(name, p, ctx, level, brace, seed, pad_shift) for name, p in cands.items()
            for ctx, level, brace, seed in batch]


def score(pool_mp, cands, batch, pad_shift):
    """cands: {name: params}; every candidate on every batch episode. Returns {name: (returns array, rows)}."""
    jobs = build_jobs(cands, batch, pad_shift)
    out = {name: [] for name in cands}
    for cand, ret, ok, reason, cls in pool_mp.imap_unordered(_job, jobs, chunksize=1):
        out[cand].append((ret, ok, reason, cls))
    return out


def centred_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def search(name, max_hours, pairs=6, sigma_b=0.2, sigma_w=0.1, lr=0.05, workers=7, seed=0, zero_every=3,
           pad_shift=True, per_class=60):
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(name=name, max_hours=max_hours, pairs=pairs, sigma_b=sigma_b, sigma_w=sigma_w, lr=lr, seed=seed,
               zero_every=zero_every, pad_shift=pad_shift, episodes_per_candidate=len(SF.CLASSES) * len(LEVELS),
               n_params=N_PARAMS, levels=LEVELS)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    pool = SF.pool("train", per_class)
    sig = np.concatenate([np.full(N_B, sigma_b), np.full(N_W, sigma_w)])
    rng = np.random.default_rng(seed)
    m = np.zeros(N_PARAMS)
    adam_m, adam_v, t_adam = np.zeros(N_PARAMS), np.zeros(N_PARAMS), 0
    log = open(out / "train.jsonl", "a")
    save_params(out / "zero.json", np.zeros(N_PARAMS))
    t0, gen = time.time(), 0
    with mp.get_context("spawn").Pool(workers) as pmp:
        while time.time() - t0 < max_hours * 3600:
            gen += 1
            tg = time.time()
            batch = make_batch(pool, gen, seed)
            eps = rng.standard_normal((pairs, N_PARAMS))
            cands = {"mean": m}
            if (gen - 1) % zero_every == 0:
                cands["zero"] = np.zeros(N_PARAMS)
            for i in range(pairs):
                cands["+%d" % i] = m + sig * eps[i]
                cands["-%d" % i] = m - sig * eps[i]
            res = score(pmp, cands, batch, pad_shift)
            fit = {k: float(np.mean([x[0] for x in v])) for k, v in res.items()}
            fp = np.array([fit["+%d" % i] for i in range(pairs)])
            fm = np.array([fit["-%d" % i] for i in range(pairs)])
            rk = centred_ranks(np.concatenate([fp, fm]))
            w = rk[:pairs] - rk[pairs:]                       # antithetic: rank(+) - rank(-)
            g = (w[:, None] * eps * sig).sum(0) / (2 * pairs * sig.mean() ** 2)   # ascent direction
            t_adam += 1
            adam_m = 0.9 * adam_m + 0.1 * g
            adam_v = 0.999 * adam_v + 0.001 * g * g
            step = lr * (adam_m / (1 - 0.9 ** t_adam)) / (np.sqrt(adam_v / (1 - 0.999 ** t_adam)) + 1e-8)
            m_new = m + step
            by_class = {}
            for ret, ok, reason, cls in res["mean"]:
                by_class.setdefault(cls, []).append(ok)
            row = {"gen": gen, "fit_mean": round(fit["mean"], 4),
                   "fit_zero": round(fit["zero"], 4) if "zero" in fit else None,
                   "paired_diff": round(fit["mean"] - fit["zero"], 4) if "zero" in fit else None,
                   "seated_mean": sum(x[1] for x in res["mean"]),
                   "seated_zero": sum(x[1] for x in res["zero"]) if "zero" in res else None,
                   "episodes": len(batch),
                   "cand_fit_mean": round(float(np.mean(np.r_[fp, fm])), 4),
                   "cand_fit_std": round(float(np.std(np.r_[fp, fm])), 4),
                   "cand_fit_max": round(float(max(np.r_[fp, fm])), 4),
                   "seated_by_class": {c: "%d/%d" % (sum(v), len(v)) for c, v in by_class.items()},
                   "norm_b": round(float(np.linalg.norm(m[:N_B])), 4), "norm_W": round(float(np.linalg.norm(m[N_B:])), 4),
                   "step_norm": round(float(np.linalg.norm(step)), 4), "gen_s": round(time.time() - tg, 1),
                   "wall_s": round(time.time() - t0, 1)}
            save_params(out / ("mean_gen%03d.json" % gen), m, gen=gen)   # the mean that was scored this generation
            m = m_new
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    save_params(out / ("mean_gen%03d.json" % (gen + 1)), m, gen=gen + 1)   # the final mean (never scored on a batch)
    return out


def select(name, workers=7, per_class=5, last=4, levels=(0, 2), braces=("none", "wriggle"), pad_shift=True):
    import eval_skill as ES
    import eval_wriggle as EW
    out = RUNS / name
    means = sorted(out.glob("mean_gen*.json"))[-last:]
    avg = np.mean([json.loads(f.read_text())["params"] for f in means], axis=0)
    save_params(out / "avg_last.json", avg, of=[f.name for f in means])
    cands = [out / "zero.json"] + means + [out / "avg_last.json"]
    ctxs = EW.contexts(["val"], per_class)
    t0 = time.time()
    rows = ES.evaluate(["es:" + str(c.resolve()) for c in cands], ctxs, levels, braces, 1, workers, progress=True,
                       pad_shift=pad_shift)
    res = []
    for c in cands:
        r = ES.summ(c.name, rows["es:" + str(c.resolve())])
        res.append(r)
        print("select: %s %d/%d" % (c.name, r["seated"], r["episodes"]), flush=True)
    best = max(res, key=lambda r: (r["seated"], -res.index(r)))     # ties go to the earlier candidate (zero first)
    (out / "best.json").write_text((out / best["ckpt"]).read_text())
    (out / "select.json").write_text(json.dumps({"best": best["ckpt"], "results": res, "levels": list(levels),
                                                 "braces": list(braces), "per_class": per_class,
                                                 "wall_s": round(time.time() - t0)}, indent=1))
    print(json.dumps(best), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="es_main")
    ap.add_argument("--max-hours", type=float, default=1.5)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--sigma-b", type=float, default=0.2)
    ap.add_argument("--sigma-w", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero-every", type=int, default=3)
    ap.add_argument("--no-pad-shift", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    if a.select:
        select(a.name, a.workers, pad_shift=not a.no_pad_shift)
    else:
        search(a.name, a.max_hours, a.pairs, a.sigma_b, a.sigma_w, a.lr, a.workers, a.seed, a.zero_every,
               not a.no_pad_shift)


if __name__ == "__main__":
    main()
