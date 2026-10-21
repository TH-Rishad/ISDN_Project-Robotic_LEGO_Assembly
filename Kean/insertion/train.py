"""Residual PPO. The actor sees frames and proprioception. The critic also sees privileged state."""

import math

import torch
from torch import nn
from Sakib.brickassembly.tasks.ppo import squash_logp

from Kean.insertion.env import ACT_DIM, PRIV_DIM
from Kean.insertion.obs import PROPRIO_F, stack_frames


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ELU(), nn.AdaptiveAvgPool2d(1))
        self.prop = nn.Sequential(nn.Linear(PROPRIO_F, 64), nn.ELU())
        self.mu = nn.Linear(80, ACT_DIM)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), math.log(0.15)))
        self.value = nn.Linear(80 + PRIV_DIM, 1)

    def features(self, frames, proprio):
        return torch.cat([self.conv(frames).flatten(1), self.prop(proprio)], -1)

    def forward(self, frames, proprio, priv=None):
        hidden = self.features(frames, proprio)
        dist = torch.distributions.Normal(self.mu(hidden), self.log_std.exp())
        value = None if priv is None else self.value(torch.cat([hidden, priv], -1)).squeeze(-1)
        return dist, value


def _batch(obs, priv, device):
    frames = torch.as_tensor(stack_frames(obs["frames"]), device=device).unsqueeze(0)
    proprio = torch.as_tensor(obs["proprio"], device=device).unsqueeze(0)
    privileged = torch.as_tensor(priv, device=device).unsqueeze(0)
    return frames, proprio, privileged


def _update(net, opt, frames, proprio, priv, actions, old_logp, rewards, dones):
    gamma, lam, clip = 0.99, 0.95, 0.2
    with torch.no_grad():
        _, values = net(frames, proprio, priv)
        adv = torch.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            nxt = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * nxt * (1.0 - dones[t]) - values[t]
            last = delta + gamma * lam * (1.0 - dones[t]) * last
            adv[t] = last
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    dist, values = net(frames, proprio, priv)
    logp = squash_logp(dist, actions)
    ratio = (logp - old_logp).exp()
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    loss = -torch.min(ratio * adv, clipped * adv).mean() + 0.5 * (values - returns).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.detach().item()


def train(env, samples=64, horizon=8, seed=0, device=None):
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    obs = env.reset()
    seen, last = 0, 0.0
    while seen < samples:
        frames, proprio, priv, actions, old_logp, rewards, dones = [], [], [], [], [], [], []
        for _ in range(horizon):
            f, p, c = _batch(obs, env.critic_vector(), dev)
            with torch.no_grad():
                dist, _ = net(f, p)
                u = dist.sample()
                logp = squash_logp(dist, u)
            obs, reward, done, _ = env.step(torch.tanh(u).squeeze(0).cpu().numpy())
            frames.append(f)
            proprio.append(p)
            priv.append(c)
            actions.append(u)
            old_logp.append(logp)
            rewards.append(reward)
            dones.append(float(done))
            seen += 1
            if done:
                obs = env.reset()
        last = _update(
            net, opt,
            torch.cat(frames), torch.cat(proprio), torch.cat(priv), torch.cat(actions), torch.cat(old_logp),
            torch.as_tensor(rewards, dtype=torch.float32, device=dev),
            torch.as_tensor(dones, dtype=torch.float32, device=dev))
    return {"loss": last, "samples": seen, "device": str(dev)}
