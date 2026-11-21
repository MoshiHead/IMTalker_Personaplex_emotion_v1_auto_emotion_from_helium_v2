#!/usr/bin/env python3
"""Verify runpod_helium_emotion_streaming.ipynb end-to-end, locally.

  1. JSON well-formed; every code cell compiles (shell cells skipped).
  2. The deploy cell embeds helium_emotion.py / integrate_helium_emotion.py
     byte-identical to emotion_head/deploy_template/, and they compile.
  3. Full chain on COPIES of the pristine repo files: the notebook's own
     Step-12 SERVER_PATCHES apply, then the embedded helium patcher patches
     the SER server (compiles, markers, idempotent), then HELIUM_SH_PATCHES
     apply to the patched start_winner_live.sh (idempotent).
  4. No hardcoded HF token leaked anywhere in the notebook.
  5. All new cells present with the expected markers; EMOTION_MODE wiring in
     the env cell; helium entries in the validation cell.
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_helium_emotion_streaming.ipynb"
SERVER_PATH = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
SH_PATH = ROOT / "IMTalker" / "start_winner_live.sh"
TEMPLATE_DIR = ROOT / "emotion_head" / "deploy_template"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"[ok] notebook JSON valid: {len(cells)} cells")


def csrc(c):
    return "".join(c["source"])


def by_id(cid):
    for c in cells:
        if c.get("id") == cid:
            return csrc(c)
    sys.exit(f"cell id={cid} not found")


def check(cond, msg):
    if not cond:
        sys.exit(f"FAIL: {msg}")


# --- 1. compile every code cell ----------------------------------------------
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    s = csrc(c)
    if s.lstrip().startswith("!"):
        continue
    try:
        compile(s, f"cell[{i}] id={c.get('id')}", "exec")
    except SyntaxError as exc:
        sys.exit(f"cell[{i}] id={c.get('id')} does not compile: {exc}")
print("[ok] all code cells compile")

# --- 2. embedded deploy modules match deploy_template -------------------------
deploy = by_id("helium-deploy")
m1 = re.search(r"HELIUM_MODULE = r'''(.*?)'''", deploy, re.DOTALL)
m2 = re.search(r"INTEGRATE_MODULE = r'''(.*?)'''", deploy, re.DOTALL)
check(m1 and m2, "deploy cell missing embedded modules")
check(m1.group(1) == (TEMPLATE_DIR / "helium_emotion.py").read_text(encoding="utf-8"),
      "embedded helium_emotion.py differs from deploy_template")
check(m2.group(1) == (TEMPLATE_DIR / "integrate_helium_emotion.py").read_text(encoding="utf-8"),
      "embedded integrate_helium_emotion.py differs from deploy_template")
compile(m1.group(1), "helium_emotion.py", "exec")
compile(m2.group(1), "integrate_helium_emotion.py", "exec")
print("[ok] embedded deploy modules match deploy_template and compile")

helium_sh_patches = json.loads(
    re.search(r"HELIUM_SH_PATCHES = json\.loads\(r'''(.*?)'''\)", deploy, re.DOTALL).group(1))
print(f"[ok] HELIUM_SH_PATCHES parse: {len(helium_sh_patches)} patches")

# Step-12 payloads live in the copied patch cell
patch_cell = by_id("emotion-stream-33")
server_patches = json.loads(
    re.search(r"SERVER_PATCHES = json\.loads\(r'''(.*?)'''\)", patch_cell, re.DOTALL).group(1))
sh_patches = json.loads(
    re.search(r"START_WINNER_PATCHES = json\.loads\(r'''(.*?)'''\)", patch_cell, re.DOTALL).group(1))
print(f"[ok] Step-12 payloads parse: {len(server_patches)} server, {len(sh_patches)} sh patches")

# --- 3. full chain on copies ---------------------------------------------------
sys.path.insert(0, str(TEMPLATE_DIR))
import integrate_helium_emotion as integ  # noqa: E402


def apply_pairs(text, pairs, name, skip_applied=True):
    for i, (anchor, repl) in enumerate(pairs, 1):
        n = text.count(anchor)
        if skip_applied and n == 0 and repl in text:
            continue
        if n != 1:
            sys.exit(f"{name} patch #{i}: anchor found {n}x (expected 1)")
        text = text.replace(anchor, repl)
    return text


tmp = Path(tempfile.mkdtemp(prefix="helium_nb_verify_"))
try:
    srv = tmp / SERVER_PATH.name
    sh = tmp / SH_PATH.name
    shutil.copy(SERVER_PATH, srv)
    shutil.copy(SH_PATH, sh)

    # Step 12 (as the notebook applies it: two apply_patches calls)
    srv.write_text(apply_pairs(srv.read_text(encoding="utf-8"), server_patches[:11], "server V1-V3"),
                   encoding="utf-8")
    srv.write_text(apply_pairs(srv.read_text(encoding="utf-8"), server_patches[11:], "server V4"),
                   encoding="utf-8")
    sh.write_text(apply_pairs(sh.read_text(encoding="utf-8"), sh_patches, "sh", ),
                  encoding="utf-8")

    # helium deploy cell: server patcher + sh patches
    check(integ.patch(srv) == 0, "helium patcher returned non-zero")
    patched = srv.read_text(encoding="utf-8")
    compile(patched, str(srv), "exec")
    for needle in ["HELIUM_EMOTION_PATCH_V2", "--helium_emotion",
                   "init_helium_emotion_manager(args)", "_hem.feed_step(",
                   "_emotion_id = _hem.current_emotion_id"]:
        check(needle in patched, f"patched server missing {needle}")
    check(integ.patch(srv) == 0, "helium patcher re-run returned non-zero")
    check(srv.read_text(encoding="utf-8") == patched, "helium patcher re-run modified the file")

    sh_text = sh.read_text(encoding="utf-8")
    sh_text = apply_pairs(sh_text, helium_sh_patches, "helium sh", skip_applied=False)
    for needle in ["HELIUM_ARGS", "--helium_emotion", '"${HELIUM_ARGS[@]}"',
                   "--helium_gate_rms", "--helium_gru_reset_silence_s"]:
        check(needle in sh_text, f"patched sh missing {needle}")
    # idempotency the way the cell implements it: HELIUM_ARGS already present -> skip
    check("HELIUM_ARGS" in sh_text, "idempotency marker missing")
    # bash syntax sanity: every exec continuation line ends with a backslash
    exec_lines = sh_text.splitlines()
    start = next(i for i, l in enumerate(exec_lines) if l.startswith("exec python"))
    j = start
    while exec_lines[j].rstrip().endswith("\\"):
        j += 1
    check(j > start + 30, "exec block looks truncated")
    print("[ok] full chain: Step-12 -> helium server patch -> sh patch (compile + markers + idempotent)")

    # helium patcher must also work on the plain server (Step 12 skipped)
    plain = tmp / ("plain_" + SERVER_PATH.name)
    shutil.copy(SERVER_PATH, plain)
    check(integ.patch(plain) == 0, "plain-profile helium patch returned non-zero")
    compile(plain.read_text(encoding="utf-8"), str(plain), "exec")
    print("[ok] helium patcher also works on the plain server (no Step 12)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --- 4. no leaked HF token ------------------------------------------------------
leak = re.findall(r"hf_[A-Za-z0-9]{25,}", NB_PATH.read_text(encoding="utf-8"))
check(not leak, f"hardcoded HF token(s) in the notebook: {leak}")
print("[ok] no hardcoded HF token in the notebook")

# --- 5. expected cells / wiring --------------------------------------------------
for cid in ["helium-params", "helium-head-fetch", "helium-deploy", "helium-head-check",
            "helium-replay", "helium-monitor", "env-overrides", "validate-patches",
            "health-wait", "success-banner", "emotion-mode-explainer", "hf-auth"]:
    by_id(cid)
env = by_id("env-overrides")
for needle in ['"AUTO_EMOTION": "1" if EMOTION_MODE == "ser" else "0"',
               '"HELIUM_EMOTION": "1" if EMOTION_MODE == "helium" else "0"',
               '"HELIUM_HEAD_PATH"', '"HELIUM_GATE_RMS"', 'env_overrides["HELIUM_EMA_ALPHA"]']:
    check(needle in env, f"env cell missing {needle}")
val = by_id("validate-patches")
for needle in ['HELIUM_EMOTION_PATCH_V2', '"${HELIUM_ARGS[@]}"', "class HeliumEmotionManager"]:
    check(needle in val, f"validation cell missing {needle}")
hl = by_id("health-wait")
for needle in ['[HELIUM-EMOTION] enabled:', "ERROR_HINTS.setdefault"]:
    check(needle in hl, f"health cell missing {needle}")
check("EMOTION_MODE" in by_id("helium-params"), "params cell missing EMOTION_MODE")
check("getpass" not in by_id("hf-auth"), "auth cell should not force an interactive prompt")
print("[ok] all new cells present with the expected wiring")

print("ALL HELIUM STREAMING NOTEBOOK CHECKS PASSED")
