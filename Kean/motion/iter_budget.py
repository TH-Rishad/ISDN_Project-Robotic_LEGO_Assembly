"""Cut only the trajectory-optimization iteration budget before warm-up."""

import os

FLOOR = 4
_SCALED = set()
_SKIP_PREFIX = ("torch", "numpy", "warp", "trimesh", "yourdfpy")
_PAIRS = (("num_iters", "inner_iters"), ("max_iterations", "inner_iterations"))


def scale_from_env():
    raw = os.environ.get("P3_ITER_SCALE", "1.0")
    try:
        scale = float(raw)
    except ValueError:
        raise SystemExit("p3: P3_ITER_SCALE=%r is not a float" % raw)
    if not 0 < scale <= 1:
        raise SystemExit("p3: P3_ITER_SCALE must be in (0, 1], got %s" % scale)
    return scale


def _is_trajopt_iter(path, name, val):
    if "trajopt" not in path:
        return False
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        return False
    if "warmup" in name or "interpolat" in name:
        return False
    return name.endswith("_iters") or name.endswith("_iterations")


def _module(obj):
    return getattr(type(obj), "__module__", "") or ""


def _skip(obj):
    return _module(obj).startswith(_SKIP_PREFIX)


def _fields(obj):
    try:
        return [(n, v) for n, v in vars(obj).items() if not n.startswith("_")]
    except TypeError:
        out = []
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            if not callable(val):
                out.append((name, val))
        return out[:80]


def _walk(obj, prefix, seen, found, depth):
    if obj is None or depth > 5 or id(obj) in seen or _skip(obj):
        return
    if isinstance(obj, (str, bytes, int, float, bool)):
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for key, val in list(obj.items())[:32]:
            _walk(val, "%s.%s" % (prefix, key), seen, found, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > 32:
            return
        for i, val in enumerate(obj):
            _walk(val, "%s[%d]" % (prefix, i), seen, found, depth + 1)
        return
    for name, val in _fields(obj):
        path = "%s.%s" % (prefix, name)
        if _is_trajopt_iter(path, name, val):
            found.append((path, obj, name, val))
        else:
            _walk(val, path, seen, found, depth + 1)


def _align(new_vals):
    for outer, inner in _PAIRS:
        if outer not in new_vals or inner not in new_vals:
            continue
        k = new_vals[inner]
        if k <= 0 or new_vals[outer] % k == 0:
            continue
        new_vals[outer] = max(k, (new_vals[outer] // k) * k)


def apply(planner, scale):
    """Scale iteration counts on `planner`. Returns (path, old, new) rows. Idempotent."""
    if id(planner) in _SCALED or scale >= 1:
        return []
    found = []
    _walk(planner, "planner", set(), found, 0)
    grouped = {}
    for path, obj, name, old in found:
        grouped.setdefault(id(obj), {"obj": obj, "fields": {}})["fields"][name] = (path, old)
    changed = []
    for item in grouped.values():
        fields = item["fields"]
        new_vals = {name: max(FLOOR, int(round(old * scale))) for name, (_, old) in fields.items()}
        _align(new_vals)
        for name, (path, old) in fields.items():
            new = new_vals[name]
            if new < old:
                setattr(item["obj"], name, new)
                changed.append((path, old, new))
    _SCALED.add(id(planner))
    return changed
