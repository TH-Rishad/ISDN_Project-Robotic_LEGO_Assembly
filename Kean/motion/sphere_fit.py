"""Repair a full-model sphere fit that misses its vertex or protrusion limit.

A fit that already meets both limits is returned unchanged, so the plan model is not refit.
"""

import numpy as np


def needs_repair(out, tol_mm, prot_mm):
    return out["vertex_max_mm"] > tol_mm + 1e-3 or out["protrusion_max_mm"] > prot_mm + 1e-3


def inward_sphere(point, normal, radius):
    """Sphere tangent to `point` from the inside. The point lies on the sphere."""
    n = np.asarray(normal, float)
    nrm = np.linalg.norm(n)
    n = n / nrm if nrm > 1e-8 else np.array([0.0, 0.0, 1.0])
    return np.asarray(point, float) - n * radius, float(radius)


def keep_within(protrusion, prot):
    return [bool(p <= prot + 1e-6) for p in protrusion]


def shrunk_radius(radius, protrusion, prot):
    """Pull a fat sphere in until its surface is inside `prot`. A sphere already inside is unchanged."""
    if protrusion <= prot + 1e-6:
        return float(radius)
    return max(1e-4, float(radius) - (float(protrusion) - float(prot)) - 1e-5)


def cover_radius(tol, prot):
    """Radius that covers the target point and stays inside the protrusion limit."""
    return max(1e-4, min(float(tol), 0.9 * float(prot)))


def _samples(comps, un, dev, n_surf):
    import torch
    import trimesh

    V = np.concatenate([np.asarray(m.vertices) for m in comps])
    VN = np.concatenate([np.asarray(m.vertex_normals) for m in comps])
    area = np.array([m.area for m in comps])
    S, SN = [], []
    for mesh, a in zip(comps, area):
        p, f = trimesh.sample.sample_surface(mesh, max(int(n_surf * a / area.sum()), 1), seed=0)
        S.append(np.asarray(p))
        SN.append(np.asarray(mesh.face_normals)[f])
    S, SN = np.concatenate(S), np.concatenate(SN)

    def exposed(pts):
        return un.sdf(torch.tensor(pts, dtype=torch.float32, device=dev)).detach().cpu().numpy() > -1e-4

    ev, es = exposed(V), exposed(S)
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    return to(V[ev]), to(VN[ev]), to(S[es]), to(SN[es])


def _protrusion(un, unit, centers, radii):
    pts = (centers[:, None, :] + radii[:, None, None] * unit[None]).reshape(-1, 3)
    return un.sdf(pts).reshape(centers.shape[0], -1).max(1).values


def _gap(pts, centers, radii):
    import torch

    if len(centers) == 0:
        return torch.full((len(pts),), 1e3, device=pts.device)
    return (torch.cdist(pts, centers) - radii[None]).min(1).values.clamp(min=0)


def _accept(un, unit, center, radius, point, tol, prot):
    import torch

    c = center.reshape(1, 3)
    r = torch.tensor([float(radius)], device=center.device, dtype=center.dtype)
    if float(torch.linalg.norm(c.reshape(3) - point.reshape(3)) - r) > tol + 1e-6:
        return None
    if float(_protrusion(un, unit, c, r)) > prot + 1e-6:
        return None
    return c.reshape(3), r.reshape(())


def _place(un, unit, point, normal, tol, prot, dev):
    import torch

    point = point.reshape(3).float()
    radius = cover_radius(tol, prot)
    hit = _accept(un, unit, point, radius, point, tol, prot)
    if hit is not None:
        return hit
    n = normal.reshape(-1).detach().float()
    nrm = torch.linalg.norm(n)
    n = n / nrm if float(nrm) > 1e-8 else torch.tensor([0.0, 0.0, 1.0], device=dev)
    for sign in (1.0, -1.0):
        for dist in np.linspace(0.0, radius + float(tol), 6):
            center = point - n * (sign * float(dist))
            hit = _accept(un, unit, center, radius, point, tol, prot)
            if hit is not None:
                return hit
    return None


def _shrink(un, unit, centers, radii, prot):
    """Reduce fat radii in place. A sphere that still sticks out at the minimum radius is dropped."""
    if len(centers) == 0:
        return centers, radii
    radii = radii.clone()
    for _ in range(4):
        excess = (_protrusion(un, unit, centers, radii) - prot).clamp(min=0)
        if float(excess.max()) <= 1e-6:
            break
        radii = (radii - excess - 1e-5).clamp(min=1e-4)
    keep = _protrusion(un, unit, centers, radii) <= prot + 1e-5
    return centers[keep], radii[keep]


def _cover(un, unit, centers, radii, pts, normals, tol, prot, dev, cap):
    import torch

    skipped = set()
    while len(centers) < cap:
        gap = _gap(pts, centers, radii)
        chosen = None
        for j in torch.argsort(gap, descending=True).tolist():
            if float(gap[j]) <= tol:
                break
            if j not in skipped:
                chosen = j
                break
        if chosen is None:
            break
        placed = _place(un, unit, pts[chosen], normals[chosen], tol, prot, dev)
        if placed is None:
            skipped.add(chosen)
            continue
        c, r = placed
        centers = torch.cat([centers, c.reshape(1, 3)])
        radii = torch.cat([radii, r.reshape(1)])
        if float(_gap(pts[chosen:chosen + 1], centers, radii)) > tol:
            centers, radii = centers[:-1], radii[:-1]
            skipped.add(chosen)
    return centers, radii, len(skipped)


def repair(probe, comps, tol, prot, dev, out, max_sph, n_surf=6000):
    """Shrink spheres that stick out, then cover any point still outside `tol`. The plan model is not touched when it already passes."""
    import torch

    tol_mm, prot_mm = tol * 1000, prot * 1000
    if not needs_repair(out, tol_mm, prot_mm):
        return out
    before = (out["vertex_max_mm"], out["protrusion_max_mm"], out["n"])
    un = probe._Union(comps, dev)
    unit = torch.tensor(probe._fib(400), dtype=torch.float32, device=dev)
    V, VN, S, SN = _samples(comps, un, dev, n_surf)
    C, R = _shrink(un, unit, out["C"], out["R"], prot)
    cap = max(max_sph, 2500)
    skipped = 0
    if len(V):
        C, R, skipped = _cover(un, unit, C, R, V, VN, tol, prot, dev, cap)
    if len(S) and len(C) < cap:
        C, R, skipped_s = _cover(un, unit, C, R, S, SN, tol, prot, dev, cap)
        skipped += skipped_s
    fixed = dict(out)
    fixed.update(C=C, R=R, n=len(C),
                 vertex_max_mm=_gap(V, C, R).max().item() * 1000 if len(V) else 0.0,
                 surface_max_mm=_gap(S, C, R).max().item() * 1000 if len(S) else 0.0,
                 protrusion_max_mm=_protrusion(un, unit, C, R).max().item() * 1000 if len(C) else 0.0)
    print("sphere repair: vertex %.2f->%.2f mm, protrusion %.2f->%.2f mm, n %d->%d, uncovered %d" % (
        before[0], fixed["vertex_max_mm"], before[1], fixed["protrusion_max_mm"], before[2], fixed["n"], skipped), flush=True)
    return fixed


def _protrusion_gate(out):
    links = out.get("links")
    if not isinstance(links, dict):
        return out
    for name, row in links.items():
        if "protrusion_max_mm" not in row:
            continue
        ok = bool(row["vertex_max_mm"] <= row["limit_mm"] + 1e-3 and row["protrusion_max_mm"] <= row["protrusion_limit_mm"] + 1e-3)
        if row.get("ok") and not ok:
            print("%s protrusion %.2f mm over %.1f  FAIL" % (name, row["protrusion_max_mm"], row["protrusion_limit_mm"]), flush=True)
        row["ok"] = ok
    if links:
        out["ok"] = bool(out.get("ok")) and all(row.get("ok") for row in links.values())
    return out


def install(probe):
    orig = probe.fit_link
    orig_profile = probe._fit_profile

    def fit_link(comps, tol, prot, dev, n_surf=6000, max_sph=600):
        out = orig(comps, tol, prot, dev, n_surf=n_surf, max_sph=max_sph)
        return repair(probe, comps, tol, prot, dev, out, max_sph, n_surf)

    def _fit_profile(prof, args):
        return _protrusion_gate(orig_profile(prof, args))

    probe.fit_link = fit_link
    probe._fit_profile = _fit_profile
    print("p3: sphere repair on; a fit already inside its limits is left unchanged", flush=True)
