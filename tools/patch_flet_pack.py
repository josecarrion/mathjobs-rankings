"""
Work around a bug in `flet pack` on Windows (flet-cli 1.1.2 and earlier).

flet_cli/__pyinstaller/win_utils.py opens the Flet executable with
open(exe_path, "ab", encoding="utf-8"), and Python refuses an encoding in
binary mode ("ValueError: binary mode doesn't take an encoding argument"),
so every Windows build fails. This script removes the encoding argument from
the installed copy. It does nothing if the line has been fixed upstream.

Run it after installing flet-cli and before running `flet pack`:
    python tools/patch_flet_pack.py
"""

import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("flet_cli")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("flet-cli is not installed; install it before running this script.")

path = Path(list(spec.submodule_search_locations)[0]) / "__pyinstaller" / "win_utils.py"
text = path.read_text(encoding="utf-8")
bad = 'open(exe_path, "ab", encoding="utf-8")'
if bad in text:
    path.write_text(text.replace(bad, 'open(exe_path, "ab")'), encoding="utf-8")
    print(f"Patched {path}")
else:
    print("flet pack bug not present; no patch needed")
