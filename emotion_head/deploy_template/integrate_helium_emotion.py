"""integrate_helium_emotion.py — idempotent patcher for the live server.

Usage (on the live machine, from the IMTalker dir):
    python integrate_helium_emotion.py <server_file.py>

Two profiles are auto-detected:
  * SER profile  — server already has the auto_emotion integration
    (anchors: init_auto_emotion_manager(args), --auto_emotion,
    _auto_emotion_mgr.current_emotion_id(...), _auto_emotion_mgr.reset_session())
  * PLAIN profile — plain AH server without SER
    (anchors: --direct_reply_hidden argparse line,
    `reply_engine = get_moshi_engine()` line,
    pending_reply_hidden.append(...), reply_engine.reset_session())

All insertions are marked HELIUM_EMOTION_PATCH_V1. A <file>.bak_helium backup
is written, the result is compile-checked, and missing REQUIRED anchors abort
without touching anything. Patching twice is a no-op.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARKER = "HELIUM_EMOTION_PATCH_V1"

IMPORT_BLOCK = [
    "",
    f"# {MARKER} — helium emotion head integration",
    "import helium_emotion as _hemo",
    "_helium_mgr = None",
]

ARGPARSE_BLOCK = [
    'parser.add_argument("--helium_emotion", action="store_true", '
    'help="Helium-hidden emotion head (replaces user-voice SER)")',
    'parser.add_argument("--helium_head_path", default="helium_emotion_head_final.pt")',
    'parser.add_argument("--helium_head_device", default="cuda")',
    'parser.add_argument("--helium_temperature", type=float, default=1.0)',
    'parser.add_argument("--helium_ema_alpha", type=float, default=0.4)',
    'parser.add_argument("--helium_enter_margin", type=float, default=0.15)',
    'parser.add_argument("--helium_min_dwell_s", type=float, default=1.5)',
]

FEED_BLOCK = [
    "if _helium_mgr is not None:",
    "    _helium_mgr.feed_step(",
    "        hidden,",
    '        ev.get("reply_codes"),',
    "        reply_pcm,",
    '        mimi=getattr(reply_engine, "mimi", None))',
]

INIT_PLAIN_BLOCK = [
    "if bool(getattr(args, 'helium_emotion', False)):",
    "    _helium_mgr = _hemo.init_helium_emotion_manager(",
    '        args, mimi=getattr(reply_engine, "mimi", None))',
]

INIT_SER_BLOCK = [
    "_helium_mgr = _hemo.init_helium_emotion_manager(args)",
]

SELECT_SER_BLOCK = [
    "if _helium_mgr is not None:",
    "    _emotion_id = _helium_mgr.current_emotion_id(default=_emotion_id)",
]

RESET_SER_BLOCK = [
    "if _helium_mgr is not None:",
    "    _helium_mgr.reset_session()",
]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _insert_after(lines, needle, block, *, required=True, all_occ=True):
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if not hits:
        return lines, 0, required
    if not all_occ:
        hits = hits[:1]
    out = list(lines)
    for i in reversed(hits):
        indent = _indent_of(out[i])
        ins = [indent + b if b.strip() else b for b in block]
        out[i + 1:i + 1] = ins
    return out, len(hits), False

def patch(server_path: Path) -> int:
    src = server_path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"[patch] {server_path.name} already patched ({MARKER} present); nothing to do.")
        return 0
    ser_profile = "init_auto_emotion_manager(args)" in src
    profile = "SER" if ser_profile else "PLAIN"
    print(f"[patch] profile: {profile}")
    lines = src.splitlines()
    failures = []

    # 1. import
    lines, n, fail = _insert_after(lines, "from renderer.models import IMTRenderer", IMPORT_BLOCK)
    if fail:
        failures.append("import anchor (from renderer.models import IMTRenderer)")
    print(f"[patch] import anchor hits: {n}")

    # 2. argparse flags
    if ser_profile:
        anchor = '"--auto_emotion"'
    else:
        anchor = '"--direct_reply_hidden"'
    lines, n, fail = _insert_after(lines, anchor, ARGPARSE_BLOCK, all_occ=False)
    if fail:
        failures.append(f"argparse anchor ({anchor})")
    print(f"[patch] argparse anchor hits: {n}")

    # 3. manager init
    if ser_profile:
        lines, n, fail = _insert_after(lines, "init_auto_emotion_manager(args)",
                                       INIT_SER_BLOCK, all_occ=False)
    else:
        lines, n, fail = _insert_after(lines, "reply_engine = get_moshi_engine()",
                                       INIT_PLAIN_BLOCK, all_occ=False)
    if fail:
        failures.append("engine-init anchor")
    print(f"[patch] init anchor hits: {n}")

    # 4. per-step feed inside the direct_reply_hidden producer branch
    lines, n, fail = _insert_after(
        lines, "pending_reply_hidden.append(hidden.squeeze(0).contiguous())", FEED_BLOCK)
    if fail:
        failures.append("feed anchor (pending_reply_hidden.append)")
    print(f"[patch] feed anchor hits: {n}")

    # 5. published emotion override / session reset
    if ser_profile:
        lines, n, fail = _insert_after(
            lines, "_auto_emotion_mgr.current_emotion_id(default=_emotion_id)", SELECT_SER_BLOCK)
        if fail:
            failures.append("selection anchor")
        print(f"[patch] selection anchor hits: {n}")
        lines, n, _ = _insert_after(lines, "_auto_emotion_mgr.reset_session()",
                                    RESET_SER_BLOCK, required=False)
        print(f"[patch] reset anchor hits: {n}")
    else:
        lines, n, _ = _insert_after(lines, "reply_engine.reset_session()",
                                    RESET_SER_BLOCK, required=False)
        print(f"[patch] reset anchor hits: {n} (plain profile)")
        print("[patch] NOTE: this server has no auto_emotion seam; the manager runs and")
        print("[patch] logs switches. Wire _helium_mgr.current_emotion_id() into your")
        print("[patch] expression controller where you consume the emotion id.")

    if failures:
        print("[patch] ABORT — missing required anchors:")
        for f in failures:
            print(f"[patch]   - {f}")
        print("[patch] no files were modified.")
        return 2

    patched = "\n".join(lines) + "\n"
    compile(patched, str(server_path), "exec")
    backup = server_path.with_suffix(server_path.suffix + ".bak_helium")
    if not backup.exists():
        shutil.copy2(server_path, backup)
    server_path.write_text(patched, encoding="utf-8")
    print(f"[patch] OK — patched {server_path.name} (backup: {backup.name})")
    print("[patch] run with: --helium_emotion --helium_head_path <helium_emotion_head_final.pt>")
    return 0


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    server = Path(argv[1])
    if not server.exists():
        print(f"[patch] not found: {server}")
        return 1
    try:
        return patch(server)
    except SyntaxError as exc:
        print(f"[patch] ABORT — patched file does not compile: {exc}")
        print("[patch] no files were modified.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

