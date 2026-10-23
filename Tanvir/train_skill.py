"""Train the skill-level wriggle policy (WRIGGLE_FIT_RL.md §6.9) with PPO on macro steps.

The environment is skill_env.WriggleSkillEnv: after a scripted approach, one decision every
10 env ticks (0.5 s) sets the 10 parameters of a wriggle primitive; theta = 0 is the scripted
controller. The actor is zero-initialised (mean theta 0) so training starts at the scripted
base. Networks, observation normalisation and the squashed-Gaussian log-prob are Sakib's
(tasks/ppo.py); the success-band curriculum (Buckets, with the --unlock-gate semantics) is
train_wriggle's. Training contexts come only from the socket family's train seeds.

    python train_skill.py --name skill_smoke --macro-steps 3000 --workers 7 --per-worker 3
    python train_skill.py --name skill_main --max-hours 3.3 --ckpt-minutes 14
Writes Tanvir/results/runs/<name>/{policy.pt, ckpt_<macro steps>.pt, train.jsonl, config.json, buckets.json}.
Selection and evaluation of the checkpoints: eval_skill.py.
"""

import argparse
import json
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

import skill_env as KE                                     # noqa: E402
import socket_env as SE                                    # noqa: E402
import socket_family as SF                                 # noqa: E402
from tasks.ppo import ActorCritic, RunningNorm, squash_logp   # noqa: E402
from train_wriggle import Buckets, BUCKETS, CLASSES, OUT   # noqa: E402


# --- workers -------------------------------------------------------------------------

def _worker(remote, n, cfg, seed):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    pool = SF.pool("train", cfg["per_class"])
    weights = np.zeros(len(BUCKETS))
    weights[0] = 1.0
    uni = cfg.get("uniform_frac", 0.0)

    def sampler(r):
        p = (1 - uni) * weights / weights.sum() + uni / len(BUCKETS)
        b = BUCKETS[r.choice(len(BUCKETS), p=p)]
        cls = CLASSES[b[0]]
        brace = list(SE.BRACE_P)[r.choice(3, p=list(SE.BRACE_P.values()))]
        return pool[cls][r.integers(len(pool[cls]))], b[1], brace

    envs = [KE.WriggleSkillEnv(sensing=cfg["sensing"], seed=seed + k, sampler=sampler, shaping=cfg["shaping"],
                               gamma=cfg["gamma"]) for k in range(n)]
    while True:
        cmd, data = remote.recv()
        if cmd == "weights":
            weights[:] = data
            remote.send(True)
        elif cmd == "reset":
            remote.send([e.reset() for e in envs])
        elif cmd == "step":
            out = []
            for e, a in zip(envs, data):
                o, r, d, info = e.step(a)
                if d:
                    o = e.reset()
                out.append((o, r, d, info if d else None))
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
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


def save(path, net, an, cn, cfg, macro_steps, buckets):
    torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "an": an.state(), "cn": cn.state(),
                "cfg": cfg, "macro_steps": macro_steps, "buckets": buckets}, path)


class SkillPolicy:
    """Deterministic macro policy from a checkpoint: actor obs -> theta = tanh(mean)."""

    def __init__(self, path):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        self.cfg = ck["cfg"]
        self.net = ActorCritic(KE.MACRO_DIM, KE.PRIV_DIM, KE.THETA_DIM)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.an = RunningNorm(KE.MACRO_DIM)
        self.an.load(ck["an"])

    def __call__(self, actor_obs):
        with torch.no_grad():
            h = self.net.trunk(torch.as_tensor(self.an(actor_obs)[None]))
            return torch.tanh(self.net.mu(h))[0].numpy().astype(float)


def train(name, workers=7, per_worker=3, horizon=50, seed=0, sensing="cheap", per_class=60, init_std=0.4,
          lr=3e-4, ent_coef=0.001, gamma=KE.GAMMA_MACRO, lam=0.95, shaping=0.5, critic_warmup=5000,
          unlock_gate=0.4, uniform_frac=0.1, ckpt_every=0, ckpt_minutes=0.0, macro_steps=10 ** 9,
          max_hours=None, std_range=(0.15, 0.6), target_kl=0.05, epochs=5, minibatches=4):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"skill": True, "sensing": sensing, "per_class": per_class, "shaping": shaping, "gamma": gamma,
           "uniform_frac": uniform_frac}
    (out / "config.json").write_text(json.dumps(dict(cfg, macro_steps=macro_steps, workers=workers,
                                                     per_worker=per_worker, horizon=horizon, seed=seed,
                                                     init_std=init_std, lr=lr, ent_coef=ent_coef, lam=lam,
                                                     critic_warmup=critic_warmup, unlock_gate=unlock_gate,
                                                     ckpt_every=ckpt_every, ckpt_minutes=ckpt_minutes,
                                                     max_hours=max_hours, std_range=std_range,
                                                     target_kl=target_kl, k_ticks=KE.K_TICKS,
                                                     macro_dim=KE.MACRO_DIM, theta_dim=KE.THETA_DIM), indent=1))
    log = open(out / "train.jsonl", "a")
    net = ActorCritic(KE.MACRO_DIM, KE.PRIV_DIM, KE.THETA_DIM, init_std=init_std)
    an, cn = RunningNorm(KE.MACRO_DIM), RunningNorm(KE.MACRO_DIM + KE.PRIV_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buckets = Buckets()
    venv = VecEnv(workers, per_worker, cfg, seed)
    venv.weights(buckets.weights())
    obs = venv.reset()
    n = venv.n
    window = deque(maxlen=200)
    t0, done_steps, update, last_ckpt_t, last_ckpt_s = time.time(), 0, 0, time.time(), 0
    save(out / "ckpt_00000000.pt", net, an, cn, cfg, 0, buckets.state())      # checkpoint 0: theta = 0, the scripted base
    while done_steps < macro_steps:
        if max_hours and time.time() - t0 > max_hours * 3600:
            print("max-hours reached", flush=True)
            break
        A = np.stack([o["actor"] for o in obs])
        Cb = np.stack([o["critic"] for o in obs])
        buf = {k: [] for k in ("a", "c", "u", "logp", "r", "d", "v")}
        infos = []
        thetas = []
        for t in range(horizon):
            an.update(A)
            cn.update(Cb)
            a_t, c_t = torch.as_tensor(an(A)), torch.as_tensor(cn(Cb))
            with torch.no_grad():
                dist, _ = net.dist(a_t)
                u = dist.sample()
                logp = squash_logp(dist, u)
                v = net.value(c_t)
            th = torch.tanh(u).numpy()
            thetas.append(th)
            res = venv.step(list(th))
            buf["a"].append(a_t)
            buf["c"].append(c_t)
            buf["u"].append(u)
            buf["logp"].append(logp)
            buf["v"].append(v)
            buf["r"].append(torch.tensor([x[1] for x in res], dtype=torch.float32))
            buf["d"].append(torch.tensor([float(x[2]) for x in res]))
            obs = [x[0] for x in res]
            A = np.stack([o["actor"] for o in obs])
            Cb = np.stack([o["critic"] for o in obs])
            for x in res:
                if x[2]:
                    infos.append(x[3])
                    buckets.record(x[3])
                    window.append(float(x[3]["success"]))
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(cn(Cb)))
        T = horizon
        R, D, V = torch.stack(buf["r"]), torch.stack(buf["d"]), torch.stack(buf["v"])
        adv = torch.zeros(T, n)
        gae = torch.zeros(n)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        flat = lambda x: x.reshape(T * n, *x.shape[2:])
        Ab, Cbb, Ub, Lb = flat(torch.stack(buf["a"])), flat(torch.stack(buf["c"])), \
            flat(torch.stack(buf["u"])), flat(torch.stack(buf["logp"]))
        advb, retb = flat(adv), flat(ret)
        advb = (advb - advb.mean()) / (advb.std() + 1e-8)
        stats, kl, actor_on = [], 0.0, done_steps >= critic_warmup
        stop = False
        for epoch in range(epochs):
            for mb in torch.randperm(T * n).chunk(minibatches):
                dist, _ = net.dist(Ab[mb])
                lp = squash_logp(dist, Ub[mb])
                ratio = (lp - Lb[mb]).exp()
                pg = -torch.min(ratio * advb[mb], ratio.clamp(0.8, 1.2) * advb[mb]).mean()
                vl = (net.value(Cbb[mb]) - retb[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + 0.5 * vl - ent_coef * ent if actor_on else 0.5 * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
                kl = float((Lb[mb] - lp).mean().detach())
                if actor_on and kl > target_kl:
                    stop = True
                    break
            if stop:
                break
        with torch.no_grad():
            net.log_std.clamp_(np.log(std_range[0]), np.log(std_range[1]))
        prev = done_steps
        done_steps += T * n
        update += 1
        buckets.update(float(np.mean(window)) if window else None, unlock_gate)
        venv.weights(buckets.weights())
        s = np.mean(stats, axis=0)
        by_class = {}
        for i in infos:
            by_class.setdefault(i["class"], []).append(i["success"])
        th_all = np.concatenate(thetas)
        row = {"update": update, "macro_steps": done_steps, "success_200": round(float(np.mean(window)), 3) if window else None,
               "episodes": len(infos), "succ_batch": round(float(np.mean([i["success"] for i in infos])), 3) if infos else None,
               "by_class": {c: round(float(np.mean(v)), 2) for c, v in by_class.items()},
               "reasons": {r: sum(i["reason"] == r for i in infos) for r in {i["reason"] for i in infos}},
               "unlocked": len(buckets.unlocked), "pg": round(s[0], 4), "vl": round(s[1], 4), "ent": round(s[2], 3),
               "kl": round(kl, 4), "stopped_early": stop, "actor_on": actor_on,
               "std": round(float(net.log_std.exp().mean().detach()), 3),
               "mean_theta": [round(float(x), 2) for x in th_all.mean(0)],
               "wall_s": round(time.time() - t0, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if update % 5 == 0:
            save(out / "policy.pt", net, an, cn, cfg, done_steps, buckets.state())
            (out / "buckets.json").write_text(json.dumps(buckets.state(), indent=1))
        due = (ckpt_every and done_steps // ckpt_every > prev // ckpt_every) or \
              (ckpt_minutes and time.time() - last_ckpt_t > ckpt_minutes * 60)
        if due:
            save(out / ("ckpt_%08d.pt" % done_steps), net, an, cn, cfg, done_steps, buckets.state())
            last_ckpt_t, last_ckpt_s = time.time(), done_steps
    if done_steps > last_ckpt_s:
        save(out / ("ckpt_%08d.pt" % done_steps), net, an, cn, cfg, done_steps, buckets.state())
    save(out / "policy.pt", net, an, cn, cfg, done_steps, buckets.state())
    (out / "buckets.json").write_text(json.dumps(buckets.state(), indent=1))
    venv.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--macro-steps", type=int, default=10 ** 9, help="stop after this many macro steps")
    ap.add_argument("--max-hours", type=float, default=None, help="wall-clock cap on training")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--per-worker", type=int, default=3, help="envs per worker process")
    ap.add_argument("--horizon", type=int, default=50, help="macro steps per env per update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensing", choices=["cheap", "ft", "none"], default="cheap")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--init-std", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=KE.GAMMA_MACRO)
    ap.add_argument("--shaping", type=float, default=0.5)
    ap.add_argument("--critic-warmup", type=int, default=5000, help="macro steps with the actor frozen")
    ap.add_argument("--unlock-gate", type=float, default=0.4)
    ap.add_argument("--uniform-frac", type=float, default=0.1, help="share of episodes drawn from all buckets")
    ap.add_argument("--ckpt-every", type=int, default=0, help="checkpoint every N macro steps")
    ap.add_argument("--ckpt-minutes", type=float, default=0.0, help="checkpoint every N minutes of wall time")
    a = ap.parse_args()
    train(a.name, a.workers, a.per_worker, a.horizon, a.seed, a.sensing, a.per_class, a.init_std, a.lr, a.ent_coef,
          a.gamma, shaping=a.shaping, critic_warmup=a.critic_warmup, unlock_gate=a.unlock_gate,
          uniform_frac=a.uniform_frac, ckpt_every=a.ckpt_every, ckpt_minutes=a.ckpt_minutes,
          macro_steps=a.macro_steps, max_hours=a.max_hours)


if __name__ == "__main__":
    main()
