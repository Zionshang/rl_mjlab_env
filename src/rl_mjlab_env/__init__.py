"""rl_mjlab_env package."""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_vendored_rsl_rl() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vendored_rsl_rl = repo_root / "rsl_rl" / "__init__.py"
    if not vendored_rsl_rl.exists():
        return

    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    loaded_rsl_rl = sys.modules.get("rsl_rl")
    loaded_path = Path(getattr(loaded_rsl_rl, "__file__", "") or "")
    if loaded_rsl_rl is not None and not loaded_path.is_relative_to(repo_root):
        for module_name in list(sys.modules):
            if module_name == "rsl_rl" or module_name.startswith("rsl_rl."):
                del sys.modules[module_name]


_prefer_vendored_rsl_rl()
