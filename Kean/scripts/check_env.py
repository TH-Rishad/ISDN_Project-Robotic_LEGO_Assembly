"""Motion import/GPU smoke check against Sakib's supplied environment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print("Python:", sys.executable, flush=True)
    try:
        from Kean.shared import load_probe
        probe = load_probe()
        if args.gpu:
            # Use the torch already imported by Sakib's probe.
            torch = probe.torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable in the supplied environment/allocation")
            print("GPU:", torch.cuda.get_device_name(0), "torch CUDA:", torch.version.cuda, flush=True)
        for name in ("full", "plan"):
            model = probe.PROFILES[name]["yml"]
            if not model.is_file():
                raise RuntimeError("missing shared collision model %s" % model)
            print("Model:", model, flush=True)
    except Exception as exc:
        print("Motion preflight failed: %s. Share this diagnostic with Sakib." % exc, file=sys.stderr)
        return 1
    print("Shared motion imports/models passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
