"""Import bootstrap for every script under scripts/<group>/.

Scripts used to live flat in scripts/ and imported shared helpers
(``v6_common``, ``_cli``, ``_paper_figstyle``) and each other
(``from run_generation import build_interventions``) by bare module name.
Importing this module puts ``src/`` and every ``scripts/<group>/`` directory
on ``sys.path`` so those imports keep working wherever a script lives.
Active groups are searched before ``scripts/unused/*``.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent

_skip = {"__pycache__", "templates", "sbatch"}
_active = sorted(p for p in SCRIPTS.iterdir() if p.is_dir() and p.name not in _skip | {"unused"})
_unused = sorted(p for p in (SCRIPTS / "unused").glob("*") if p.is_dir() and p.name not in _skip)
for _p in reversed([REPO / "src", *_active, *_unused]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
