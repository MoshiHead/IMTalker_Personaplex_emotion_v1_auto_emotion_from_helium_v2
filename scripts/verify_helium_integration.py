#!/usr/bin/env python3
"""Verify the HELIUM_EMOTION_PATCH_V2 deploy integration end-to-end, locally.

  1. SER profile: apply the old notebook's V3 SERVER_PATCHES to a copy of the
     pristine server, then run integrate_helium_emotion.patch() on it.
     The patched server must compile and contain all helium markers.
  2. Idempotency: patching twice is a no-op; upgrading from a V1-patched file
     restores the .bak_helium backup and reproduces the identical V2 result.
  3. PLAIN profile: patch a copy of the pristine server directly; must compile.
  4. helium_emotion.py deploy module: the manager's gating / GRU-reset /
     config.json auto-tuning behave as designed (dummy head, CPU).
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
SERVER_PATH = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
DEPLOY_DIR = ROOT / "emotion_head" / "deploy_template"

sys.path.insert(0, str(DEPLOY_DIR))
sys.path.insert(0, str(ROOT / "emotion_head"))
import integrate_helium_emotion as integ  # noqa: E402
import helium_emotion as hemo  # noqa: E402
import eh_common as eh  # noqa: E402

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
c33 = next("".join(c["source"]) for c in nb["cells"] if c.get("id") == "emotion-stream-33")
server_patches = json.loads(
    re.search(r"SERVER_PATCHES = json\.loads\(r'''(.*?)'''\)", c33, re.DOTALL).group(1))
print(f"[ok] extracted {len(server_patches)} V3 server patches from the old notebook")


def apply_v3(path):
    text = path.read_text(encoding="utf-8")
    for i, (anchor, repl) in enumerate(server_patches, 1):
        n = text.count(anchor)
        if n == 0 and repl in text:
            print(f"[ok] V3 patch #{i} already applied in the local server copy; skipping")
            continue
        if n != 1:
            sys.exit(f"V3 anchor #{i} found {n}x in pristine server")
        text = text.replace(anchor, repl)
    path.write_text(text, encoding="utf-8")


def check(cond, msg):
    if not cond:
        sys.exit(f"FAIL: {msg}")


tmp = Path(tempfile.mkdtemp(prefix="helium_verify_"))
try:
    # --- 1. SER profile ------------------------------------------------------
    srv = tmp / SERVER_PATH.name
    shutil.copy(SERVER_PATH, srv)
    apply_v3(srv)
    check(integ.patch(srv) == 0, "SER-profile patch returned non-zero")
    patched = srv.read_text(encoding="utf-8")
    compile(patched, str(srv), "exec")
    for needle in ["HELIUM_EMOTION_PATCH_V2", '"--helium_emotion"',
                   '"--helium_gate_rms"', '"--helium_gru_reset_silence_s"',
                   "_hemo.init_helium_emotion_manager(args)",
                   '_hem.feed_step(hidden, ev.get("reply_codes"), reply_pcm,',
                   "_emotion_id = _hem.current_emotion_id(default=_emotion_id)",
                   "_hem.reset_session()"]:
        check(needle in patched, f"SER-patched server missing: {needle}")
    # init must anchor on the CALL inside build_app (4-space indent), not the stub
    check("        init_auto_emotion_manager(args)\n    # HELIUM_EMOTION_PATCH_V2" in patched,
          "helium init block not inserted after the init_auto_emotion_manager CALL")
    # selection override must sit OUTSIDE the SER None-check (8-space indent)
    check("            _emotion_id = _auto_emotion_mgr.current_emotion_id(default=_emotion_id)\n"
          "        _hem = _hemo.get_helium_emotion_manager()" in patched,
          "helium select block is not outside the SER None-check")
    # feed block must sit after pending_reply_hidden.append (28-space indent)
    check("                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())\n"
          "                            _hem = _hemo.get_helium_emotion_manager()" in patched,
          "helium feed block not inserted after pending_reply_hidden.append")
    print("[ok] SER profile: patches apply, compile, markers + indentation correct")

    # --- 2. idempotency + legacy upgrade -------------------------------------
    check(integ.patch(srv) == 0, "second patch returned non-zero")
    check(srv.read_text(encoding="utf-8") == patched, "second patch modified the file")
    v1ish = patched.replace("HELIUM_EMOTION_PATCH_V2", "HELIUM_EMOTION_PATCH_V1")
    srv.write_text(v1ish, encoding="utf-8")
    check(integ.patch(srv) == 0, "V1-upgrade patch returned non-zero")
    repatched = srv.read_text(encoding="utf-8")
    check(repatched == patched, "V1->V2 re-patch differs from the first V2 result")
    compile(repatched, str(srv), "exec")
    print("[ok] idempotency + legacy V1 restore path")

    # --- 3. PLAIN profile -----------------------------------------------------
    plain = tmp / ("plain_" + SERVER_PATH.name)
    shutil.copy(SERVER_PATH, plain)
    check(integ.patch(plain) == 0, "PLAIN-profile patch returned non-zero")
    psrc = plain.read_text(encoding="utf-8")
    compile(psrc, str(plain), "exec")
    for needle in ["HELIUM_EMOTION_PATCH_V2", '"--helium_emotion"',
                   "_hemo.init_helium_emotion_manager(args)",
                   '_hem.feed_step(hidden, ev.get("reply_codes"), reply_pcm,']:
        check(needle in psrc, f"PLAIN-patched server missing: {needle}")
    check("_emotion_id = _hem.current_emotion_id" not in psrc,
          "PLAIN profile must not get a selection override (no seam)")
    check(integ.patch(plain) == 0, "second PLAIN patch returned non-zero")
    check(plain.read_text(encoding="utf-8") == psrc, "second PLAIN patch modified the file")
    print("[ok] PLAIN profile: patches apply, compile, idempotent")

    # --- 4. manager behaviour (gating / GRU reset / config.json) --------------
    import os
    log_dir = tmp / "logs"
    os.environ["IMTALKER_EMOTION_LOG_DIR"] = str(log_dir)
    head = eh.EmotionFusionHead()
    hp = tmp / "helium_emotion_head_final.pt"
    eh.save_head(head, str(hp), extra={})
    (tmp / "config.json").write_text(json.dumps(
        {"smoother": {"ema_alpha": 0.9, "enter_margin": 0.33, "min_dwell_s": 2.5}}))
    args = type("A", (), {
        "emotion_id": 0, "helium_head_path": str(hp), "helium_head_device": "cpu",
        "helium_temperature": 1.0, "helium_ema_alpha": -1.0, "helium_enter_margin": -1.0,
        "helium_min_dwell_s": -1.0, "helium_gate_rms": 0.006,
        "helium_gru_reset_silence_s": 0.5})()
    mgr = hemo.HeliumEmotionManager(args)
    # config.json must override the hardcoded smoother defaults
    check(abs(mgr.smoother.ema_alpha - 0.9) < 1e-9, "config.json ema_alpha not picked up")
    check(abs(mgr.smoother.enter_margin - 0.33) < 1e-9, "config.json enter_margin not picked up")
    check(mgr.smoother.min_dwell_frames == int(2.5 * 12.5), "config.json min_dwell_s not picked up")
    # explicit CLI value beats config.json
    args2 = type("A", (), {})()
    args2.emotion_id = 0; args2.helium_head_path = str(hp)
    args2.helium_head_device = "cpu"; args2.helium_temperature = 1.0
    args2.helium_ema_alpha = 0.5; args2.helium_enter_margin = -1.0
    args2.helium_min_dwell_s = -1.0; args2.helium_gate_rms = 0.006
    args2.helium_gru_reset_silence_s = 3.0
    mgr2 = hemo.HeliumEmotionManager(args2)
    check(abs(mgr2.smoother.ema_alpha - 0.5) < 1e-9, "CLI ema_alpha must override config.json")
    mgr.start()
    check(len(list(log_dir.glob("helium_emotion_*.csv"))) == 1, "CSV log not created")
    loud = (np.random.randn(1920) * 0.1).astype("float32")
    silent = np.zeros(1920, dtype="float32")
    for _ in range(10):
        mgr.feed_step(torch.randn(1, 4096), None, silent)
    check(mgr.gated_steps == 0, "silence frames must not advance the smoother")
    check(mgr.steps_fed == 10, "silent steps not counted")
    check(mgr._runner.steps == 0 and mgr._runner._h is None,
          "GRU state not reset after >= reset_silence_s of silence")
    for _ in range(30):
        mgr.feed_step(torch.randn(1, 4096), None, loud)
    check(mgr.gated_steps == 30, "loud frames must advance the smoother")
    check(mgr._runner.steps == 30, "GRU must step on speech frames")
    check(mgr.feed_errors == 0, "feed errors must be zero")
    mgr.stop()
    csv_text = next(log_dir.glob("helium_emotion_*.csv")).read_text()
    check("timestamp,step,raw_emotion,emotion,confidence,switched" in csv_text,
          "CSV header missing")
    print("[ok] manager: gating, GRU silence-reset, config.json tuning, CSV log")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("ALL HELIUM INTEGRATION CHECKS PASSED")
