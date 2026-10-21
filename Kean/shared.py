"""Single boundary for loading Sakib's existing cell/probe without copying it.

GPU dependencies are lazy: importing Kean and running CPU tests needs no Newton.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

KEAN = Path(__file__).resolve().parent
ROOT = KEAN.parent
CELL = ROOT / "Sakib" / "brickassembly"
PROBE = CELL / "scripts" / "12_curobo_probe.py"


def add_cell_path():
    path = str(CELL)
    if path not in sys.path:
        sys.path.insert(0, path)


def configure_probe(probe):
    """Redirect experiment reports only; Sakib's model paths remain authoritative."""
    probe.OUT = KEAN / "results" / "p3"
    return probe


def load_probe():
    """Load once, before warmup; top-level motion remains Sakib's package."""
    name = "Kean._curobo_probe"
    if name in sys.modules:
        return sys.modules[name]
    add_cell_path()
    spec = importlib.util.spec_from_file_location(name, PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        configure_probe(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module
