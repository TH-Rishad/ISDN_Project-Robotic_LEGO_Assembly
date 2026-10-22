"""Train the wriggle-fit policy (WRIGGLE_FIT_RL.md §3.4-§3.6) in the MuJoCo twin.

  1. Behaviour cloning: run the scripted controller (ScriptedWriggle) on the training family,
     keep the episodes it seats, and regress the actor's mean onto its actions.
  2. PPO from that actor, end-to-end (default) or residual, with the success-band curriculum:
     buckets are (socket class, aim-error level); a bucket unlocks when its easier neighbour
     (one class or one level down) seats >= 50 %; unlocked buckets are drawn with weight 1
     while their success is in [0.2, 0.8] (or they have < 20 episodes), else 0.2. With
     --unlock-gate g, nothing new unlocks while the last 200 episodes seat below g.

Residual safeguards (WRIGGLE_FIT_RL.md §6.4-§6.5): --res-scale caps the residual at that
fraction of the action range, --no-res-release leaves release to the scripted base, --res-l2
penalises the actor's mean residual, --critic-warmup trains the critic alone for that many
samples first, and a checkpoint is kept every --ckpt-every samples (ckpt_<samples>.pt, the
zero-residual start included); --select then evaluates them on the family's validation split
and copies the best to best.pt.

Networks, observation normalisation and the squashed-Gaussian log-prob are Sakib's
(tasks/ppo.py: ActorCritic, RunningNorm, squash_logp); the environments are
socket_env.SocketInsertionEnv in worker processes. Training contexts come only from the
socket family's train seeds; the repo's assemblies stay held out (eval_wriggle.py).

    python train_wriggle.py --name smoke --samples 20000 --workers 4 --bc-episodes 40
    python train_wriggle.py --name main --mode residual --samples 1500000 --workers 7   # the main run (~hours)
    python train_wriggle.py --name r2 --mode residual --safe --samples 300000 --select   # with the safeguards
    python train_wriggle.py --name ft --sensing ft ...                         # ablations
Writes Tanvir/results/runs/<name>/{policy.pt, train.jsonl, config.json, ckpt_*.pt, best.pt, select.json}.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402
import torch.nn as nn                                      # noqa: E402

import socket_env as SE                                    # noqa: E402
import socket_family as SF                                 # noqa: E402
from tasks.ppo import ActorCritic, RunningNorm, squash_logp   # noqa: E402

OUT = HERE / "results" / "runs"
CLASSES = SF.CLASSES
N_LEVELS = len(SE.ERR_LEVELS)
BUCKETS = [(c, l) for c in range(len(CLASSES)) for l in range(N_LEVELS)]


# --- workers -------------------------------------------------------------------------

def _worker(remote, n, cfg, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    pool = SF.pool("train", cfg["per_class"])
    weights = np.zeros(len(BUCKETS))
    weights[0] = 1.0

    def sampler(r):
        b = BUCKETS[r.choice(len(BUCKETS), p=weights / weights.sum())]
        cls = CLASSES[b[0]]
        brace = list(SE.BRACE_P)[r.choice(3, p=list(SE.BRACE_P.values()))]
        return pool[cls][r.integers(len(pool[cls]))], b[1], brace

    envs = [SE.SocketInsertionEnv(mode=cfg["mode"], sensing=cfg["sensing"], seed=seed + k, sampler=sampler,
                                  res_scale=cfg.get("res_scale", 1.0), res_release=cfg.get("res_release", True),
                                  shaping=cfg.get("shaping", 0.0))
            for k in range(n)]
    obs = None
    while True:
        cmd, data = remote.recv()
        if cmd == "weights":
            weights[:] = data
            remote.send(True)
        elif cmd == "reset":
            obs = [e.reset() for e in envs]
            remote.send(obs)
        elif cmd == "step":
            out = []
            for e, a in zip(envs, data):
                o, r, d, info = e.step(a)
                base = e.last_base.copy()
                if d:
                    info = dict(info)
                    o = e.reset()
                out.append((o, r, d, info if d else None, base))
            remote.send(out)
        elif cmd == "close":
            remote.close()
            return


class VecEnv:
    def __init__(self, workers, per_worker, cfg, seed):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, per_worker, cfg, seed + 1000 * w), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.per, self.n = per_worker, workers * per_worker

    def _all(self, cmd, data=None):
        for r in self.remotes:
            r.send((cmd, data))
        return [r.recv() for r in self.remotes]

    def weights(self, w):
        self._all("weights", w)

    def reset(self):
        return [o for chunk in self._all("reset") for o in chunk]

    def step(self, actions):
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[w * self.per:(w + 1) * self.per]))
        return [x for r in self.remotes for x in r.recv()]

    def close(self):
        self._all("close") if False else [r.send(("close", None)) for r in self.remotes]
        for p in self.procs:
            p.join(timeout=5)


# --- curriculum ------------------------------------------------------------------------

class Buckets:
    """Success-band sampling over (class, level) buckets with frontier unlocking."""

    def __init__(self, window=50):
        self.hist = {b: deque(maxlen=window) for b in BUCKETS}
        self.unlocked = {BUCKETS[0]}

    def record(self, info):
        b = (CLASSES.index(info["class"]), info["level"])
        self.hist[b].append(float(info["success"]))

    def rate(self, b):
        h = self.hist[b]
        return float(np.mean(h)) if h else None

    def update(self, overall=None, gate=0.0):
        if overall is not None and overall < gate:
            return
        for c, l in BUCKETS:
            if (c, l) in self.unlocked:
                continue
            for e in ((c - 1, l), (c, l - 1)):
                if e in self.unlocked and len(self.hist[e]) >= 20 and self.rate(e) >= 0.5:
                    self.unlocked.add((c, l))
                    break

    def weights(self):
        w = np.zeros(len(BUCKETS))
        for n, b in enumerate(BUCKETS):
            if b in self.unlocked:
                r = self.rate(b)
                w[n] = 1.0 if r is None or len(self.hist[b]) < 20 or 0.2 <= r <= 0.8 else 0.2
        return w

    def state(self):
        return {"%s/%d" % (CLASSES[c], l): {"n": len(self.hist[(c, l)]), "rate": self.rate((c, l)),
                                             "unlocked": (c, l) in self.unlocked} for c, l in BUCKETS}


# --- behaviour cloning -------------------------------------------------------------------

def behaviour_clone(net, an, cfg, episodes, workers, seed, dev, log):
    """Scripted episodes on buckets up to aim-error level 2, all classes; regress on the seated ones."""
    vcfg = dict(cfg, mode="scripted")
    venv = VecEnv(workers, 1, vcfg, seed + 77)
    w = np.array([1.0 if l <= 2 else 0.0 for _, l in BUCKETS])
    venv.weights(w)
    obs = venv.reset()
    ep_obs = [[] for _ in range(venv.n)]
    ep_act = [[] for _ in range(venv.n)]
    X, Y, done_eps, kept = [], [], 0, 0
    t0 = time.time()
    while done_eps < episodes:
        res = venv.step([np.zeros(SE.ACT_DIM)] * venv.n)
        for k, (o, r, d, info, base) in enumerate(res):
            ep_obs[k].append(obs[k]["actor"])
            ep_act[k].append(base)
            if d:
                done_eps += 1
                if info["success"]:
                    kept += 1
                    X += ep_obs[k]
                    Y += ep_act[k]
                ep_obs[k], ep_act[k] = [], []
        obs = [x[0] for x in res]
    venv.close()
    print("BC: %d scripted episodes, %d seated, %d samples, %.0fs" % (done_eps, kept, len(X), time.time() - t0),
          flush=True)
    if not X:
        return
    X = np.array(X)
    an.update(X)
    Xt = torch.as_tensor(an(X), device=dev)
    U = torch.as_tensor(np.arctanh(np.clip(np.array(Y), -0.97, 0.97)), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(list(net.trunk.parameters()) + list(net.mu.parameters()), lr=1e-3)
    for epoch in range(30):
        perm = torch.randperm(len(Xt), device=dev)
        tot = 0.0
        for mb in perm.split(256):
            loss = (net.mu(net.trunk(Xt[mb])) - U[mb]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(mb)
    log.write(json.dumps({"bc_episodes": done_eps, "bc_seated": kept, "bc_samples": len(X),
                          "bc_loss": round(tot / len(Xt), 4)}) + "\n")
    print("BC loss %.4f" % (tot / len(Xt)), flush=True)


# --- PPO -------------------------------------------------------------------------------------

def _save(path, net, an, cn, cfg, samples, buckets):
    torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "an": an.state(),
                "cn": cn.state(), "cfg": cfg, "samples": samples, "buckets": buckets}, path)


def train(name, samples, workers=7, horizon=128, seed=0, mode="e2e", sensing="cheap", bc_episodes=300,
          per_class=60, device="cpu", init_std=0.25, res_scale=1.0, res_release=True, res_l2=0.0,
          critic_warmup=0, unlock_gate=0.0, ckpt_every=0, shaping=0.0):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dev = torch.device(device)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"mode": mode, "sensing": sensing, "per_class": per_class, "res_scale": res_scale,
           "res_release": res_release, "shaping": shaping}
    (out / "config.json").write_text(json.dumps(dict(cfg, samples=samples, workers=workers, horizon=horizon,
                                                     seed=seed, bc_episodes=bc_episodes, init_std=init_std,
                                                     res_l2=res_l2, critic_warmup=critic_warmup,
                                                     unlock_gate=unlock_gate, ckpt_every=ckpt_every), indent=1))
    log = open(out / "train.jsonl", "a")
    net = ActorCritic(SE.OBS_DIM, SE.PRIV_DIM, SE.ACT_DIM, init_std=init_std).to(dev)
    an, cn = RunningNorm(SE.OBS_DIM), RunningNorm(SE.OBS_DIM + SE.PRIV_DIM)
    if bc_episodes and mode == "e2e":
        behaviour_clone(net, an, cfg, bc_episodes, workers, seed, dev, log)
    if samples <= 0:                                    # BC only: save the cloned actor and stop
        _save(out / "policy.pt", net, an, cn, cfg, 0, {})
        return out / "policy.pt"
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    buckets = Buckets()
    venv = VecEnv(workers, 1, cfg, seed)
    venv.weights(buckets.weights())
    obs = venv.reset()
    n = venv.n
    window = deque(maxlen=200)
    t0, done_samples, update = time.time(), 0, 0
    if ckpt_every:
        _save(out / "ckpt_0000000.pt", net, an, cn, cfg, 0, buckets.state())
    while done_samples < samples:
        A = np.stack([o["actor"] for o in obs])
        Cb = np.stack([o["critic"] for o in obs])
        buf = {k: [] for k in ("a", "c", "u", "logp", "r", "d", "v")}
        infos = []
        for t in range(horizon):
            an.update(A)
            cn.update(Cb)
            a_t, c_t = torch.as_tensor(an(A), device=dev), torch.as_tensor(cn(Cb), device=dev)
            with torch.no_grad():
                dist, _ = net.dist(a_t)
                u = dist.sample()
                logp = squash_logp(dist, u)
                v = net.value(c_t)
            res = venv.step(list(torch.tanh(u).cpu().numpy()))
            buf["a"].append(a_t)
            buf["c"].append(c_t)
            buf["u"].append(u)
            buf["logp"].append(logp)
            buf["v"].append(v)
            buf["r"].append(torch.tensor([x[1] for x in res], dtype=torch.float32, device=dev))
            buf["d"].append(torch.tensor([float(x[2]) for x in res], device=dev))
            obs = [x[0] for x in res]
            A = np.stack([o["actor"] for o in obs])
            Cb = np.stack([o["critic"] for o in obs])
            for x in res:
                if x[2]:
                    infos.append(x[3])
                    buckets.record(x[3])
                    window.append(float(x[3]["success"]))
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(cn(Cb), device=dev))
        T = horizon
        R, D, V = torch.stack(buf["r"]), torch.stack(buf["d"]), torch.stack(buf["v"])
        adv = torch.zeros(T, n, device=dev)
        gae = torch.zeros(n, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + 0.99 * nv * (1 - D[t]) - V[t]
            gae = delta + 0.99 * 0.95 * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        flat = lambda x: x.reshape(T * n, *x.shape[2:])
        Ab, Cbb, Ub, Lb = flat(torch.stack(buf["a"])), flat(torch.stack(buf["c"])), \
            flat(torch.stack(buf["u"])), flat(torch.stack(buf["logp"]))
        advb, retb = flat(adv), flat(ret)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        stats = []
        actor_on = done_samples >= critic_warmup
        for epoch in range(5):
            for mb in torch.randperm(T * n, device=dev).chunk(4):
                dist, _ = net.dist(Ab[mb])
                lp = squash_logp(dist, Ub[mb])
                ratio = (lp - Lb[mb]).exp()
                pg = -torch.min(ratio * advb[mb], ratio.clamp(0.8, 1.2) * advb[mb]).mean()
                vl = (net.value(Cbb[mb]) - retb[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                l2 = dist.mean.pow(2).sum(-1).mean()
                loss = pg + 0.5 * vl - 0.001 * ent + res_l2 * l2 if actor_on else 0.5 * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item(), l2.item()))
        prev = done_samples
        done_samples += T * n
        update += 1
        buckets.update(float(np.mean(window)) if window else None, unlock_gate)
        venv.weights(buckets.weights())
        s = np.mean(stats, axis=0)
        by_class = {}
        for i in infos:
            by_class.setdefault(i["class"], []).append(i["success"])
        row = {"update": update, "samples": done_samples, "success_200": round(float(np.mean(window)), 3) if window else None,
               "episodes": len(infos),
               "by_class": {c: round(float(np.mean(v)), 2) for c, v in by_class.items()},
               "reasons": {r: sum(i["reason"] == r for i in infos) for r in {i["reason"] for i in infos}},
               "unlocked": len(buckets.unlocked), "pg": round(s[0], 4), "vl": round(s[1], 4),
               "ent": round(s[2], 3), "res_l2": round(s[3], 4), "actor_on": actor_on, "std": round(float(net.log_std.exp().mean().detach()), 3),
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if update % 10 == 0 or done_samples >= samples:
            _save(out / "policy.pt", net, an, cn, cfg, done_samples, buckets.state())
            (out / "buckets.json").write_text(json.dumps(buckets.state(), indent=1))
        if ckpt_every and (done_samples // ckpt_every > prev // ckpt_every or done_samples >= samples):
            _save(out / ("ckpt_%07d.pt" % done_samples), net, an, cn, cfg, done_samples, buckets.state())
    venv.close()
    return out / "policy.pt"


def select_best(out, workers=7, per_class=3, levels=(0, 2), braces=("none", "wriggle"), seeds=1):
    """Evaluate every ckpt_*.pt in out on the family validation split; copy the best to best.pt.
    Ties go to the earlier checkpoint (closer to the scripted base)."""
    import shutil
    import eval_wriggle as EW
    ctxs = EW.contexts(["val"], per_class)
    res = []
    for ck in sorted(out.glob("ckpt_*.pt")):
        rows = EW.evaluate(str(ck.resolve()), ctxs, levels, braces, seeds, workers)
        res.append({"ckpt": ck.name, "seated": sum(r["success"] for r in rows), "episodes": len(rows),
                    "joint_break": sum(r["reason"] == "joint_break" for r in rows)})
        print("select: %s %d/%d" % (ck.name, res[-1]["seated"], len(rows)), flush=True)
    best = max(res, key=lambda r: (r["seated"], -res.index(r)))
    shutil.copy(out / best["ckpt"], out / "best.pt")
    (out / "select.json").write_text(json.dumps({"best": best["ckpt"], "results": res, "levels": list(levels),
                                                 "braces": list(braces), "per_class": per_class}, indent=1))
    return best


class Policy:
    """Deterministic policy from a checkpoint: actor obs -> action (tanh of the mean)."""

    def __init__(self, path):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        self.cfg = ck["cfg"]
        self.net = ActorCritic(SE.OBS_DIM, SE.PRIV_DIM, SE.ACT_DIM)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.an = RunningNorm(SE.OBS_DIM)
        self.an.load(ck["an"])

    def __call__(self, actor_obs):
        with torch.no_grad():
            h = self.net.trunk(torch.as_tensor(self.an(actor_obs)[None]))
            return torch.tanh(self.net.mu(h))[0].numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["e2e", "residual"], default="e2e")
    ap.add_argument("--sensing", choices=["cheap", "ft", "none"], default="cheap")
    ap.add_argument("--bc-episodes", type=int, default=300)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init-std", type=float, default=None,
                    help="initial exploration std (default 0.15 residual, as Sakib's R1-R4; 0.25 e2e)")
    ap.add_argument("--safe", action="store_true",
                    help="residual safeguards: --res-scale 0.3 --no-res-release --res-l2 0.1 "
                         "--critic-warmup 20000 --unlock-gate 0.4 --ckpt-every 100000")
    ap.add_argument("--res-scale", type=float, default=None)
    ap.add_argument("--no-res-release", action="store_true")
    ap.add_argument("--res-l2", type=float, default=None)
    ap.add_argument("--critic-warmup", type=int, default=None)
    ap.add_argument("--unlock-gate", type=float, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="weight of the potential-based depth shaping (WRIGGLE_FIT_RL.md §3.5); 0 = off")
    ap.add_argument("--select", action="store_true", help="after training, pick the best checkpoint on validation")
    a = ap.parse_args()
    std = a.init_std if a.init_std is not None else (0.15 if a.mode == "residual" else 0.25)
    pick = lambda v, safe, off: v if v is not None else (safe if a.safe else off)
    out = train(a.name, a.samples, a.workers, a.horizon, a.seed, a.mode, a.sensing, a.bc_episodes, a.per_class,
                a.device, std, res_scale=pick(a.res_scale, 0.3, 1.0),
                res_release=not (a.no_res_release or a.safe), res_l2=pick(a.res_l2, 0.1, 0.0),
                critic_warmup=pick(a.critic_warmup, 20_000, 0), unlock_gate=pick(a.unlock_gate, 0.4, 0.0),
                ckpt_every=pick(a.ckpt_every, 100_000, 0), shaping=a.shaping)
    if a.select:
        print(json.dumps(select_best(out.parent, a.workers)), flush=True)


if __name__ == "__main__":
    main()
