"""integrate_helium_emotion.py — idempotent exact-anchor patcher for the live server.

Usage (on the live machine, from the IMTalker dir):
    python integrate_helium_emotion.py <server_file.py>

Two profiles are auto-detected:
  * SER profile  — server already carries the auto_emotion integration
    (user-voice SER; the default live flow: the streaming notebook's
    emotion-injection step runs FIRST, then this patcher).
  * PLAIN profile — plain AH server without SER.

Every patch is an exact (anchor -> replacement) string substitution; each
anchor must occur EXACTLY once or the patcher aborts WITHOUT touching the
file. All insertions are marked HELIUM_EMOTION_PATCH_V2, a
<file>.bak_helium backup is written, and the patched file is compile-checked.
Patching twice is a no-op; upgrading from HELIUM_EMOTION_PATCH_V1 restores the
backup and re-patches.

What gets wired (SER profile; PLAIN omits `select` — no emotion seam):
  import   guarded `import helium_emotion as _hemo`
  args     --helium_emotion / --helium_head_path / --helium_head_device /
           --helium_temperature / --helium_ema_alpha / --helium_enter_margin /
           --helium_min_dwell_s / --helium_gate_rms / --helium_gru_reset_silence_s
  init     manager created whenever --helium_emotion is set (INDEPENDENT of
           --auto_emotion; anchored on the CALL `init_auto_emotion_manager(args)`,
           never on the fallback stub `def`)
  feed     per-80ms-step head update inside the direct_reply_hidden producer
           (hidden [1,4096], reply_codes [1,8,1], reply_pcm [1920], live mimi)
  select   helium prediction wins over the SER emotion id
  reset    manager reset per browser session

Run with: --helium_emotion --helium_head_path helium_emotion_head_final.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "HELIUM_EMOTION_PATCH_V2"
LEGACY_MARKERS = ("HELIUM_EMOTION_PATCH_V1",)

# ---------------------------------------------------------------------------
# Inserted blocks (indentation matches their anchors exactly)
# ---------------------------------------------------------------------------
ARGPARSE_BLOCK = (
    "        # HELIUM_EMOTION_PATCH_V2 (agent emotion from Helium hidden states)\n"
    '        parser.add_argument("--helium_emotion", action="store_true",\n'
    '                            help="Agent-emotion head from Helium hidden states (replaces user-voice SER)")\n'
    '        parser.add_argument("--helium_head_path", default="helium_emotion_head_final.pt")\n'
    '        parser.add_argument("--helium_head_device", default="cuda")\n'
    '        parser.add_argument("--helium_temperature", type=float, default=1.0)\n'
    '        parser.add_argument("--helium_ema_alpha", type=float, default=-1.0,\n'
    '                            help="-1 = use config.json next to the head (training grid-search)")\n'
    '        parser.add_argument("--helium_enter_margin", type=float, default=-1.0)\n'
    '        parser.add_argument("--helium_min_dwell_s", type=float, default=-1.0)\n'
    '        parser.add_argument("--helium_gate_rms", type=float, default=0.006,\n'
    '                            help="smoother only advances while reply RMS >= this (assistant speaking)")\n'
    '        parser.add_argument("--helium_gru_reset_silence_s", type=float, default=3.0,\n'
    '                            help="reset GRU context after this much assistant silence; <=0 disables")\n'
)

INIT_BLOCK = (
    "    # HELIUM_EMOTION_PATCH_V2: agent-emotion head (independent of --auto_emotion)\n"
    '    if bool(getattr(args, "helium_emotion", False)):\n'
    "        if _hemo is not None:\n"
    "            _hemo.init_helium_emotion_manager(args)\n"
    "        else:\n"
    '            print("[HELIUM-EMOTION] disabled: helium_emotion.py not deployed", flush=True)\n'
)

FEED_ANCHOR = "                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())\n"
FEED_BLOCK = (
    "                            _hem = _hemo.get_helium_emotion_manager() if _hemo is not None else None\n"
    "                            if _hem is not None:\n"
    "                                _hem.feed_step(hidden, ev.get(\"reply_codes\"), reply_pcm,\n"
    "                                               mimi=getattr(reply_engine, \"mimi\", None))\n"
)

SELECT_BLOCK = (
    "        _hem = _hemo.get_helium_emotion_manager() if _hemo is not None else None\n"
    "        if _hem is not None:\n"
    "            _emotion_id = _hem.current_emotion_id(default=_emotion_id)\n"
)

RESET_BLOCK = (
    "        _hem = _hemo.get_helium_emotion_manager() if _hemo is not None else None\n"
    "        if _hem is not None:\n"
    "            _hem.reset_session()\n"
)

# ---------------------------------------------------------------------------
# Anchors (exact strings from the live server / the SER-patched server)
# ---------------------------------------------------------------------------
SER_IMPORT_ANCHOR = "    from auto_emotion import get_auto_emotion_manager, init_auto_emotion_manager\n"
SER_IMPORT_BLOCK = (
    "    # HELIUM_EMOTION_PATCH_V2\n"
    "    try:\n"
    "        import helium_emotion as _hemo\n"
    "    except Exception:\n"
    "        _hemo = None\n"
)
PLAIN_IMPORT_ANCHOR = "from renderer.models import IMTRenderer\n"
PLAIN_IMPORT_BLOCK = (
    "# HELIUM_EMOTION_PATCH_V2\n"
    "try:\n"
    "    import helium_emotion as _hemo\n"
    "except Exception:\n"
    "    _hemo = None\n"
)

SER_ARGPARSE_ANCHOR = (
    '        parser.add_argument("--auto_emotion", action="store_true",\n'
    '                            help="Detect the user\'s emotion from their voice each utterance '
    'and drive the avatar\'s expression automatically (--emotion_id becomes the '
    'initial/fallback emotion)")\n'
)
PLAIN_ARGPARSE_ANCHOR = (
    '        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, '
    'help="Use Moshi generation hidden directly instead of re-encoding reply audio")\n'
)

SER_INIT_ANCHOR = (
    '    if bool(getattr(args, "auto_emotion", False)):\n'
    "        init_auto_emotion_manager(args)\n"
)
PLAIN_INIT_ANCHOR = (
    "        fm_engine = get_engine()\n"
    "        fm_engine.reset_session()\n"
)
PLAIN_INIT_BLOCK = (
    "        # HELIUM_EMOTION_PATCH_V2: agent-emotion head (created once per process)\n"
    '        if bool(getattr(args, "helium_emotion", False)) and _hemo is not None \\\n'
    "                and _hemo.get_helium_emotion_manager() is None:\n"
    "            _hemo.init_helium_emotion_manager(args)\n"
)

SER_SELECT_ANCHOR = (
    "        _auto_emotion_mgr = get_auto_emotion_manager()\n"
    "        if _auto_emotion_mgr is not None:\n"
    "            _emotion_id = _auto_emotion_mgr.current_emotion_id(default=_emotion_id)\n"
)

SER_RESET_ANCHOR = (
    "        _auto_emotion_mgr = get_auto_emotion_manager()\n"
    "        if _auto_emotion_mgr is not None:\n"
    "            _auto_emotion_mgr.reset_session()\n"
)
PLAIN_RESET_ANCHOR = (
    "        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None\n"
    "        if reply_engine is not None:\n"
    "            reply_engine.reset_session()\n"
)


def _ser_patches():
    return [
        ("import", SER_IMPORT_ANCHOR, SER_IMPORT_ANCHOR + SER_IMPORT_BLOCK),
        ("argparse", SER_ARGPARSE_ANCHOR, SER_ARGPARSE_ANCHOR + ARGPARSE_BLOCK),
        ("init", SER_INIT_ANCHOR, SER_INIT_ANCHOR + INIT_BLOCK),
        ("feed", FEED_ANCHOR, FEED_ANCHOR + FEED_BLOCK),
        ("select", SER_SELECT_ANCHOR, SER_SELECT_ANCHOR + SELECT_BLOCK),
        ("reset", SER_RESET_ANCHOR, SER_RESET_ANCHOR + RESET_BLOCK),
    ]


def _plain_patches():
    return [
        ("import", PLAIN_IMPORT_ANCHOR, PLAIN_IMPORT_ANCHOR + PLAIN_IMPORT_BLOCK),
        ("argparse", PLAIN_ARGPARSE_ANCHOR, PLAIN_ARGPARSE_ANCHOR + ARGPARSE_BLOCK),
        ("init", PLAIN_INIT_ANCHOR, PLAIN_INIT_ANCHOR + PLAIN_INIT_BLOCK),
        ("feed", FEED_ANCHOR, FEED_ANCHOR + FEED_BLOCK),
        ("reset", PLAIN_RESET_ANCHOR, PLAIN_RESET_ANCHOR + RESET_BLOCK),
    ]
def patch(server_path: Path) -> int:
    src = server_path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"[patch] {server_path.name} already patched ({MARKER} present); nothing to do.")
        return 0
    backup = server_path.with_suffix(server_path.suffix + ".bak_helium")
    if any(m in src for m in LEGACY_MARKERS):
        if backup.exists():
            print(f"[patch] legacy patch found; restoring {backup.name} before re-patching")
            src = backup.read_text(encoding="utf-8")
        else:
            print("[patch] ABORT — legacy HELIUM_EMOTION_PATCH_V1 present but no "
                  ".bak_helium backup to restore from; revert the file manually.")
            return 3

    ser_profile = "init_auto_emotion_manager(args)" in src
    profile = "SER" if ser_profile else "PLAIN"
    print(f"[patch] profile: {profile}")
    patches = _ser_patches() if ser_profile else _plain_patches()

    failures = []
    for name, anchor, repl in patches:
        n = src.count(anchor)
        if n != 1:
            failures.append(f"{name} anchor found {n}x (expected exactly 1)")
            print(f"[patch]   {name}: anchor count {n} — FAIL")
        else:
            src = src.replace(anchor, repl)
            print(f"[patch]   {name}: patched")

    if not ser_profile:
        print("[patch] NOTE: plain profile has no auto_emotion seam; the helium manager")
        print("[patch] runs and logs switches, but _emotion_id is not overridden — wire")
        print("[patch] _hemo.get_helium_emotion_manager().current_emotion_id() into your")
        print("[patch] expression controller where you consume the emotion id.")

    if failures:
        print("[patch] ABORT — anchor problems (the upstream repo changed?):")
        for f in failures:
            print(f"[patch]   - {f}")
        print("[patch] no files were modified.")
        return 2

    try:
        compile(src, str(server_path), "exec")
    except SyntaxError as exc:
        print(f"[patch] ABORT — patched file does not compile: {exc}")
        print("[patch] no files were modified.")
        return 3

    if not backup.exists():
        backup.write_text(server_path.read_text(encoding="utf-8"), encoding="utf-8")
    server_path.write_text(src, encoding="utf-8")
    print(f"[patch] OK — patched {server_path.name} (backup: {backup.name})")
    print("[patch] run with: --helium_emotion --helium_head_path helium_emotion_head_final.pt")
    print("[patch] (wire the matching HELIUM_ARGS into start_winner_live.sh — the")
    print("[patch]  streaming notebook's helium step does this automatically)")
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

