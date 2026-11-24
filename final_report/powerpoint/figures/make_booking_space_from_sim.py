"""Booking-space figures matching Kean.motion.reservations (what really runs).

A booking is NOT one full-arm cuboid and NOT the exact swept-sphere volume.
It is a set of per-link axis-aligned boxes: for each link, the AABB of that
link's collision spheres along the dense joint path. Straight Cartesian moves
additionally inflate those boxes by 1.5× hand travel (+ turn term).

Data: dual_pyramid_s3_studfit_031 recording.npz + fr3_plan.yml spheres + the
run's Newton 8× MP4 for overlays.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIG = ROOT / "final_report/figures"
VID = ROOT / "final_report/videos"
RUN = ROOT / "Kean/results/dual_pyramid_s3_studfit_031"
SPHERE_YML = ROOT / "Sakib/brickassembly/motion/fr3_plan.yml"
SIM_MP4 = RUN / "dual_pyramid_s3_studfit_031_8x.mp4"

CAM_POS = np.array([0.0, -0.8, 1.55], float)
CAM_PITCH, CAM_YAW = -62.0, 90.0

ARM_BASE = {
    "A": (np.array([-0.45, 0.0, 0.0]), 0.0),
    "B": (np.array([0.45, 0.0, 0.0]), math.pi),
}

# Panda/FR3 modified DH (m).
_MDH = [
    (0.0, 0.333, 0.0, 0),
    (0.0, 0.0, -math.pi / 2, 1),
    (0.0, 0.316, math.pi / 2, 2),
    (0.0825, 0.0, math.pi / 2, 3),
    (-0.0825, 0.384, -math.pi / 2, 4),
    (0.0, 0.0, math.pi / 2, 5),
    (0.088, 0.0, math.pi / 2, 6),
    (0.0, 0.107, 0.0, -1),
]
LINK_FRAME = {
    "fr3_link1": 1,
    "fr3_link2": 2,
    "fr3_link3": 3,
    "fr3_link4": 4,
    "fr3_link5": 5,
    "fr3_link6": 6,
    "fr3_link7": 7,
    "fr3_hand": 8,
    "fr3_leftfinger": 8,
    "fr3_rightfinger": 8,
}
# Draw order / emphasis (distal links matter most for the slide).
LINK_ORDER = [
    "fr3_link1",
    "fr3_link2",
    "fr3_link3",
    "fr3_link4",
    "fr3_link5",
    "fr3_link6",
    "fr3_link7",
    "fr3_hand",
    "fr3_leftfinger",
    "fr3_rightfinger",
]
FINGER_LOCK = 0.0075


def yaw_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mdh(a, d, alpha, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fr3_frames(q7):
    T = np.eye(4)
    out = [T.copy()]
    for a, d, alpha, j in _MDH:
        T = T @ mdh(a, d, alpha, float(q7[j]) if j >= 0 else 0.0)
        out.append(T.copy())
    return out


def from_base(arm, pts):
    o, yaw = ARM_BASE[arm]
    return np.asarray(pts, float) @ yaw_R(yaw).T + o


def load_spheres(yml, stride=2):
    raw = yaml.safe_load(yml.read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
    by_link = {}
    for link in LINK_ORDER:
        items = raw.get(link) or []
        pts = [[*s["center"], s["radius"]] for i, s in enumerate(items) if i % stride == 0]
        if pts:
            by_link[link] = np.asarray(pts, float)
    return by_link


def arm_link_spheres(arm, q7, by_link, grip=0.04):
    """Dict link -> (N,4) world spheres at one configuration."""
    frames = fr3_frames(q7)
    R_hand = frames[LINK_FRAME["fr3_hand"]][:3, :3]
    y = R_hand[:, 1]
    out = {}
    for link, local in by_link.items():
        T = frames[LINK_FRAME[link]]
        c = (T[:3, :3] @ local[:, :3].T).T + T[:3, 3]
        if link == "fr3_leftfinger":
            c = c + (grip - FINGER_LOCK) * y
        elif link == "fr3_rightfinger":
            c = c - (grip - FINGER_LOCK) * y
        out[link] = np.c_[from_base(arm, c), local[:, 3]]
    return out


def tcp_world(arm, q7):
    T = fr3_frames(q7)[8]
    p = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.1034])
    return from_base(arm, p[None])[0]


def per_link_booking(link_sphere_seq, pad=0.0):
    """Match reservations.bounds: per-link AABB of swept spheres, optional Cartesian pad."""
    links = [L for L in LINK_ORDER if L in link_sphere_seq[0]]
    lows, highs = [], []
    for link in links:
        stack = np.concatenate([frame[link] for frame in link_sphere_seq], axis=0)
        lo = (stack[:, :3] - stack[:, 3:4]).min(axis=0) - pad
        hi = (stack[:, :3] + stack[:, 3:4]).max(axis=0) + pad
        lows.append(lo)
        highs.append(hi)
    return links, np.asarray(lows), np.asarray(highs)


def separated(lo_a, hi_a, lo_b, hi_b, margin=0.025):
    """reservations.separated: True if no per-link box pair overlaps within margin."""
    overlap = np.all(
        (lo_a[:, None] <= hi_b[None] + margin) & (lo_b[None] <= hi_a[:, None] + margin), axis=2
    )
    return not bool(overlap.any())


def box_corners(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )


def add_aabb(ax, lo, hi, color, alpha=0.08, lw=1.2):
    c = box_corners(lo, hi)
    faces = [
        [c[i] for i in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (4, 7, 3, 0),
        )
    ]
    ax.add_collection3d(
        Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=lw, alpha=alpha)
    )


def draw_spheres(ax, spheres, color, alpha=0.35, max_n=60):
    if len(spheres) == 0:
        return
    idx = np.linspace(0, len(spheres) - 1, min(max_n, len(spheres))).astype(int)
    u, v = np.linspace(0, 2 * np.pi, 8), np.linspace(0, np.pi, 6)
    for x, y, z, r in spheres[idx]:
        xs = x + r * np.outer(np.cos(u), np.sin(v))
        ys = y + r * np.outer(np.sin(u), np.sin(v))
        zs = z + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, shade=False)


def world_to_pixel(pts, w, h, pos=CAM_POS, pitch_deg=CAM_PITCH, yaw_deg=CAM_YAW, fl=1.15):
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    forward = np.array(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    )
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = np.asarray(pts, float) - pos
    x, y, z = rel @ right, rel @ up, np.maximum(rel @ forward, 1e-3)
    return np.c_[w * 0.5 + (fl * h * 0.5) * (x / z), h * 0.5 - (fl * h * 0.5) * (y / z)], z


def draw_boxes_2d(img, lows, highs, color, thickness=1.5, skip_proximal=2):
    """Project per-link AABBs; skip the first few proximal links to reduce clutter."""
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for lo, hi in zip(lows[skip_proximal:], highs[skip_proximal:]):
        uv, z = world_to_pixel(box_corners(lo, hi), w, h)
        for a, b in edges:
            if z[a] > 0.05 and z[b] > 0.05:
                ax.plot(
                    [uv[a, 0], uv[b, 0]],
                    [uv[a, 1], uv[b, 1]],
                    color=color,
                    lw=thickness,
                    solid_capstyle="round",
                    alpha=0.9,
                )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgba


def load_run():
    manifest = json.loads((RUN / "manifest.json").read_text())
    with np.load(RUN / manifest["recording"], allow_pickle=True) as z:
        rec = {k: z[k] for k in z.files}
    return manifest, rec


def segment_indices(rec, t0, t1, step=8):
    t = rec["sim_time"]
    i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
    return list(range(i0, max(i0 + 1, i1), step))


def equal_aspect(ax, pts):
    c = pts.mean(axis=0)
    r = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.08
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(max(0.0, c[2] - r), c[2] + r)
    return c, r


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    print("loading spheres + recording…", flush=True)
    by_link = load_spheres(SPHERE_YML, stride=2)
    manifest, rec = load_run()
    q = rec["q_exec"]

    # Free-space transport: both arms moving — booking = per-link swept AABBs (no Cartesian pad).
    t0, t1 = 8.0, 14.0
    idxs = segment_indices(rec, t0, t1, step=10)
    print(f"segment t={t0}-{t1}s  n={len(idxs)}  {manifest['names']}", flush=True)

    seq_A, seq_B, tcp_A, tcp_B = [], [], [], []
    for i in idxs:
        seq_A.append(arm_link_spheres("A", q[i, :7], by_link, float(q[i, 7])))
        seq_B.append(arm_link_spheres("B", q[i, 9:16], by_link, float(q[i, 16])))
        tcp_A.append(tcp_world("A", q[i, :7]))
        tcp_B.append(tcp_world("B", q[i, 9:16]))
    tcp_A, tcp_B = np.asarray(tcp_A), np.asarray(tcp_B)
    travel_A = float(np.linalg.norm(tcp_A[-1] - tcp_A[0]))
    travel_B = float(np.linalg.norm(tcp_B[-1] - tcp_B[0]))

    links_A, lo_A, hi_A = per_link_booking(seq_A, pad=0.0)
    links_B, lo_B, hi_B = per_link_booking(seq_B, pad=0.0)
    # Straight-move rule illustration (not applied on this free-space segment).
    _, lo_A_cart, hi_A_cart = per_link_booking(seq_A, pad=1.5 * travel_A)
    ok = separated(lo_A, hi_A, lo_B, hi_B, margin=0.025)
    print(
        f"travel A={travel_A*1e3:.0f} mm  B={travel_B*1e3:.0f} mm  "
        f"per-link bookings separated (≥25 mm)? {ok}",
        flush=True,
    )

    mid_A = seq_A[len(seq_A) // 2]
    mid_B = seq_B[len(seq_B) // 2]
    all_mid = np.vstack([*mid_A.values(), *mid_B.values()])[:, :3]
    extent = np.vstack([lo_A, hi_A, lo_B, hi_B])

    # ---- 3D still: per-link boxes + current spheres + trajectories ----
    fig = plt.figure(figsize=(11.5, 7), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        "Booking space as implemented  ·  per-link AABBs of swept spheres\n"
        "dual_pyramid_s3_studfit_031  ·  sim t = %.0f–%.0f s (free-space transport)"
        % (t0, t1),
        fontsize=11,
    )
    for link, sp in mid_A.items():
        draw_spheres(ax, sp, "#3b82f6", alpha=0.28, max_n=40)
    for link, sp in mid_B.items():
        draw_spheres(ax, sp, "#fb923c", alpha=0.28, max_n=40)
    for lo, hi in zip(lo_A, hi_A):
        add_aabb(ax, lo, hi, "#2563eb", alpha=0.07, lw=1.3)
    for lo, hi in zip(lo_B, hi_B):
        add_aabb(ax, lo, hi, "#c2410c", alpha=0.07, lw=1.3)
    ax.plot(tcp_A[:, 0], tcp_A[:, 1], tcp_A[:, 2], color="#1d4ed8", lw=2.6, label="A hand trajectory")
    ax.plot(tcp_B[:, 0], tcp_B[:, 1], tcp_B[:, 2], color="#9a3412", lw=2.6, label="B hand trajectory")
    ax.plot([], [], [], color="#2563eb", lw=3, label="A booking (one AABB per link)")
    ax.plot([], [], [], color="#c2410c", lw=3, label="B booking (one AABB per link)")
    for name, col in (("A", "#1e3a8a"), ("B", "#9a3412")):
        o, _ = ARM_BASE[name]
        ax.scatter(*o, c=col, s=36)
        ax.text(o[0], o[1], o[2] + 0.04, name, color=col, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=28, azim=-70)
    equal_aspect(ax, extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.text(
        0.5,
        0.015,
        "reservations.bounds: for each link, AABB of that link’s spheres along the path. "
        "Not one full-arm cuboid; not the raw swept volume.  separated() requires ≥ 25 mm between any link-box pair.",
        ha="center",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout()
    still = OUT / "booking_space_sim.png"
    fig.savefig(still, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", still, flush=True)

    # ---- slide: top-down, distal links emphasized ----
    fig = plt.figure(figsize=(11.5, 6.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        "Booking space  ·  per-link swept AABBs (as in Kean.motion.reservations)",
        fontsize=12,
    )
    for sp in mid_A.values():
        draw_spheres(ax, sp, "#3b82f6", alpha=0.32, max_n=35)
    for sp in mid_B.values():
        draw_spheres(ax, sp, "#fb923c", alpha=0.32, max_n=35)
    # emphasize distal half of links
    n = len(lo_A)
    for i, (lo, hi) in enumerate(zip(lo_A, hi_A)):
        add_aabb(ax, lo, hi, "#2563eb", alpha=0.05 if i < n // 2 else 0.12, lw=1.0 if i < n // 2 else 1.8)
    for i, (lo, hi) in enumerate(zip(lo_B, hi_B)):
        add_aabb(ax, lo, hi, "#ea580c", alpha=0.05 if i < n // 2 else 0.12, lw=1.0 if i < n // 2 else 1.8)
    ax.plot(tcp_A[:, 0], tcp_A[:, 1], tcp_A[:, 2], color="#1d4ed8", lw=2.8)
    ax.plot(tcp_B[:, 0], tcp_B[:, 1], tcp_B[:, 2], color="#9a3412", lw=2.8)
    ax.text(*tcp_A[len(tcp_A) // 2], "A trajectory", color="#1d4ed8", fontsize=8)
    ax.text(*tcp_B[len(tcp_B) // 2], "B trajectory", color="#9a3412", fontsize=8)
    ax.text(*(0.5 * (lo_A[-1] + hi_A[-1])), "A link bookings", color="#1d4ed8", fontsize=9)
    ax.text(*(0.5 * (lo_B[-1] + hi_B[-1])), "B link bookings", color="#9a3412", fontsize=9)
    status = "bookings OK (≥ 25 mm)" if ok else "would wait (overlap within 25 mm)"
    ax.text2D(
        0.02,
        0.02,
        "Free-space move: pad = 0.   Straight move would expand each box by 1.5× hand travel.   " + status,
        transform=ax.transAxes,
        fontsize=8,
        color="#111827",
    )
    ax.view_init(elev=62, azim=-90)
    equal_aspect(ax, extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.tight_layout()
    slide = OUT / "booking_space_slide.png"
    fig.savefig(slide, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", slide, flush=True)

    # ---- inset panel: one arm’s links, showing boxes ≠ raw sweep ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=160, subplot_kw={"projection": "3d"})
    for ax, seq, lo, hi, tcp, color, title in (
        (axes[0], seq_A, lo_A, hi_A, tcp_A, "#2563eb", "Arm A — swept spheres vs per-link boxes"),
        (axes[1], seq_B, lo_B, hi_B, tcp_B, "#c2410c", "Arm B — swept spheres vs per-link boxes"),
    ):
        # ghost spheres along the path (hand + link7 only, to show sweep)
        for fr in seq[::3]:
            for link in ("fr3_link7", "fr3_hand", "fr3_leftfinger", "fr3_rightfinger"):
                if link in fr:
                    draw_spheres(ax, fr[link], color, alpha=0.12, max_n=20)
        for box_lo, box_hi in zip(lo, hi):
            add_aabb(ax, box_lo, box_hi, color, alpha=0.08, lw=1.2)
        ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=color, lw=2.2)
        ax.set_title(title, fontsize=10)
        equal_aspect(ax, np.vstack([lo, hi]))
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(
        "What is booked: the axis-aligned box of each link’s swept spheres (not the sphere sausage itself)",
        fontsize=11,
    )
    fig.tight_layout()
    explain = OUT / "booking_space_per_link.png"
    fig.savefig(explain, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", explain, flush=True)

    # ---- GIF ----
    fig = plt.figure(figsize=(10, 6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    c, r = equal_aspect(ax, extent)

    def update(fi):
        ax.cla()
        ax.set_title("Per-link bookings stay fixed  ·  spheres move inside their link boxes", fontsize=11)
        for lo, hi in zip(lo_A, hi_A):
            add_aabb(ax, lo, hi, "#2563eb", alpha=0.06, lw=1.1)
        for lo, hi in zip(lo_B, hi_B):
            add_aabb(ax, lo, hi, "#ea580c", alpha=0.06, lw=1.1)
        for sp in seq_A[fi].values():
            draw_spheres(ax, sp, "#3b82f6", alpha=0.4, max_n=30)
        for sp in seq_B[fi].values():
            draw_spheres(ax, sp, "#fb923c", alpha=0.4, max_n=30)
        ax.plot(tcp_A[: fi + 1, 0], tcp_A[: fi + 1, 1], tcp_A[: fi + 1, 2], color="#1d4ed8", lw=2)
        ax.plot(tcp_B[: fi + 1, 0], tcp_B[: fi + 1, 1], tcp_B[: fi + 1, 2], color="#9a3412", lw=2)
        ax.plot(tcp_A[:, 0], tcp_A[:, 1], tcp_A[:, 2], color="#93c5fd", lw=1, ls=":")
        ax.plot(tcp_B[:, 0], tcp_B[:, 1], tcp_B[:, 2], color="#fdba74", lw=1, ls=":")
        ax.view_init(elev=28, azim=-70)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(max(0, c[2] - r), c[2] + r)
        return []

    anim = FuncAnimation(fig, update, frames=len(seq_A), interval=120)
    gif = OUT / "booking_space.gif"
    anim.save(gif, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("wrote", gif, flush=True)

    # ---- Overlay on real Newton MP4 ----
    print("overlay on", SIM_MP4.name, flush=True)
    reader = imageio.get_reader(SIM_MP4)
    fps = float(reader.get_meta_data().get("fps", 30) or 30)
    picks = np.linspace(int(t0 / 8 * fps), int(t1 / 8 * fps), 24).astype(int)
    frames = []
    for vi in picks:
        frame = reader.get_data(int(vi))
        img = draw_boxes_2d(frame, lo_A, hi_A, "#3b82f6", thickness=1.4, skip_proximal=3)
        img = draw_boxes_2d(img, lo_B, hi_B, "#f97316", thickness=1.4, skip_proximal=3)
        h, w = img.shape[:2]
        uv_a, za = world_to_pixel(tcp_A, w, h)
        uv_b, zb = world_to_pixel(tcp_B, w, h)
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.plot(uv_a[za > 0.05, 0], uv_a[za > 0.05, 1], color="#60a5fa", lw=2.4)
        ax.plot(uv_b[zb > 0.05, 0], uv_b[zb > 0.05, 1], color="#fdba74", lw=2.4)
        ax.text(
            20,
            36,
            "dual_pyramid_s3_studfit_031  ·  per-link booking boxes (as in reservations.py)",
            color="white",
            fontsize=11,
            bbox=dict(facecolor="black", alpha=0.55, pad=4),
        )
        ax.text(
            20,
            70,
            "each wire box = one link’s swept-sphere AABB   ·   not one full-arm cuboid",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.45, pad=3),
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    reader.close()

    overlay = OUT / "booking_space_sim_overlay.png"
    imageio.imwrite(overlay, frames[len(frames) // 2])
    mp4 = OUT / "booking_space_sim.mp4"
    imageio.mimwrite(mp4, frames, fps=8, codec="libx264", quality=7)
    print("wrote", overlay, flush=True)
    print("wrote", mp4, flush=True)

    for src, dst_dir in (
        (still, FIG),
        (slide, FIG),
        (explain, FIG),
        (overlay, FIG),
        (gif, VID),
        (mp4, VID),
    ):
        shutil.copy(src, dst_dir / src.name)
    print("copied to final_report/figures and final_report/videos", flush=True)


if __name__ == "__main__":
    main()
