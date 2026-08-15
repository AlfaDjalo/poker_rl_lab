#!/usr/bin/env python3
"""
migrate_to_package.py — one-time repo restructure into an installable
`poker_rl_lab` package.

Run from the repo root:

    python3 migrate_to_package.py

What it does
------------
1. Moves these top-level RL-Lab packages/modules under poker_rl_lab/:
     actions/ envs/ representation/ scenarios/ models/ trainers/
     evaluation/ controllers/ experiments/ mock_poker_eval.py
2. Adds __init__.py to poker_rl_lab/ and any of the above dirs that are
   missing one.
3. Rewrites internal absolute imports across the whole repo so that
   `from actions.x import Y` / `import actions.x` (and the same for the
   other 8 top-level names + mock_poker_eval) become
   `from poker_rl_lab.actions.x import Y` / `import poker_rl_lab.actions.x`.

Does NOT touch:
  - poker_engine/*  (separate Engine component/dependency)
  - top-level run_*.py demo scripts (left importing poker_rl_lab.* after
    rewrite, no relocation needed since they're just entry points)

Safe to re-run: skips the move if poker_rl_lab/ already exists; import
rewriting is idempotent (re-prefixing an already-prefixed import is a
no-op match).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "poker_rl_lab"

MOVED_PACKAGES = [
    "actions", "envs", "representation", "scenarios",
    "models", "trainers", "evaluation", "controllers", "experiments",
]
MOVED_MODULES = ["mock_poker_eval"]  # single-file top-level modules

ALL_MOVED_NAMES = MOVED_PACKAGES + MOVED_MODULES


def move_into_package() -> None:
    PKG.mkdir(exist_ok=True)
    (PKG / "__init__.py").touch(exist_ok=True)

    for name in MOVED_PACKAGES:
        src = ROOT / name
        if not src.is_dir():
            print(f"  [skip] {name}/ not found at repo root")
            continue
        dst = PKG / name
        if dst.exists():
            print(f"  [skip] poker_rl_lab/{name}/ already exists")
            continue
        shutil.move(str(src), str(dst))
        init = dst / "__init__.py"
        if not init.exists():
            init.touch()
        print(f"  [moved] {name}/ -> poker_rl_lab/{name}/")

    for name in MOVED_MODULES:
        src = ROOT / f"{name}.py"
        if not src.is_file():
            print(f"  [skip] {name}.py not found at repo root")
            continue
        dst = PKG / f"{name}.py"
        if dst.exists():
            print(f"  [skip] poker_rl_lab/{name}.py already exists")
            continue
        shutil.move(str(src), str(dst))
        print(f"  [moved] {name}.py -> poker_rl_lab/{name}.py")


# Matches `from actions...` / `from actions.x.y import ...`
_FROM_RE = re.compile(
    r"^(\s*from\s+)(" + "|".join(ALL_MOVED_NAMES) + r")(\.|(?=\s+import))",
)
# Matches `import actions` / `import actions.x.y`
_IMPORT_RE = re.compile(
    r"^(\s*import\s+)(" + "|".join(ALL_MOVED_NAMES) + r")(\.|(?=\s|$|,))",
)


def rewrite_imports_in_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    out_lines = []
    for line in text.splitlines(keepends=True):
        new_line = _FROM_RE.sub(r"\1poker_rl_lab.\2\3", line)
        new_line = _IMPORT_RE.sub(r"\1poker_rl_lab.\2\3", new_line)
        if new_line != line:
            changed = True
        out_lines.append(new_line)
    if changed:
        path.write_text("".join(out_lines))
    return changed


def rewrite_all_imports() -> None:
    py_files = [
        p for p in ROOT.rglob("*.py")
        if "poker_engine" not in p.parts  # never touch the Engine component
    ]
    n_changed = 0
    for p in py_files:
        try:
            if rewrite_imports_in_file(p):
                n_changed += 1
                print(f"  [rewrote imports] {p.relative_to(ROOT)}")
        except Exception:
            print(f"FAILED: {p}")
            raise                
    print(f"\nRewrote imports in {n_changed} file(s).")


if __name__ == "__main__":
    print("Step 1: moving packages into poker_rl_lab/ ...")
    move_into_package()
    print("\nStep 2: rewriting internal imports ...")
    rewrite_all_imports()
    print("\nDone. Next: pip install -e .")