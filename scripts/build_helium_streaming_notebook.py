#!/usr/bin/env python3
"""Build runpod_helium_emotion_streaming.ipynb from runpod_emotion_streaming.ipynb.

Strategy: copy every working cell of the proven SER notebook verbatim and
splice in the helium additions (title, helium params, token-free HF auth,
head fetch, helium deploy patch cell, head/replay checks, updated
env/health/READY cells, helium monitor). Every code cell is AST-validated at
build time; the embedded deploy modules are checked byte-identical to
emotion_head/deploy_template/.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "runpod_emotion_streaming.ipynb"
OUT_NB = ROOT / "runpod_helium_emotion_streaming.ipynb"
TEMPLATE_DIR = ROOT / "emotion_head" / "deploy_template"

HELIUM_MODULE = (TEMPLATE_DIR / "helium_emotion.py").read_text(encoding="utf-8")
INTEGRATE_MODULE = (TEMPLATE_DIR / "integrate_helium_emotion.py").read_text(encoding="utf-8")
assert "'''" not in HELIUM_MODULE and "'''" not in INTEGRATE_MODULE, "cannot r'''-embed"


def code(cell_id, src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(cell_id, src):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }

# ---------------------------------------------------------------------------
# New / replaced cell texts
# ---------------------------------------------------------------------------
TITLE_MD = """\
# RunPod — EmotionPlex AH streaming server (HELIUM agent-emotion edition)

Same proven live stack as `runpod_emotion_streaming.ipynb` (IMTalker + PersonaPlex
+ emotion LoRA + silence emotion CFG + exact-anchor server patches), plus the
**helium agent-emotion head**: the avatar's expression now follows the
**assistant's generated speech**, predicted every 80 ms from the Helium
`transformer_out` hidden state + mimi reply latents + reply prosody by the GRU
head trained in notebooks 02-05 (`helium_emotion_head_final.pt`, 2.03M params).

`EMOTION_MODE` in the parameters area selects the behaviour:

* `"helium"` (default) — agent-emotion head drives the expression. The user-voice
  SER is skipped entirely (no funasr install, no SER model download, no
  `[AUTO-EMOTION]` runtime). Live parity with training: the smoother only advances
  while the assistant is speaking (reply-RMS gate 0.006) and the GRU context
  resets after 3 s of assistant silence.
* `"ser"` — original behaviour (emotion2vec SER on the user's mic).
* `"fixed"` — `EMOTION` stays fixed; no auto switching at all.

Prerequisites on this pod:

1. This notebook is the same one-time setup flow as the SER notebook — run it
   top to bottom; steps that are already done are no-ops.
2. The trained head + `config.json`: either set `HF_HELIUM_REPO` to the repo you
   uploaded from training notebook 06 (the fetch step downloads it), or copy
   `helium_emotion_head_final.pt` + `config.json` into `/workspace/IMTalker/`
   manually and set `DOWNLOAD_HELIUM_HEAD = False`.
3. Optional: the `emotion_head` feature dir on this pod (same pod as training)
   enables the offline replay sanity test — it compares the deploy smoother
   against the Phase-B test split without starting the server.

Stop any other kernels/processes using the GPU before starting the server
(a training kernel holding VRAM will OOM it).
"""
HELIUM_PARAMS = """\
from pathlib import Path

# --- HELIUM AGENT-EMOTION (this notebook's addition) -------------------------
# EMOTION_MODE: "helium" = avatar expression follows the ASSISTANT's generated
# speech via the trained helium-hidden GRU head. "ser" = old behaviour (SER on
# the USER's mic). "fixed" = EMOTION stays fixed (no auto switching).
EMOTION_MODE = "helium"
AUTO_EMOTION = (EMOTION_MODE == "ser")   # gates the SER install/download/run cells

# Hugging Face repo with the trained head + config (uploaded by training
# notebook 06), or copy the files to HELIUM_HEAD_PATH / HELIUM_CONFIG_PATH
# manually and set DOWNLOAD_HELIUM_HEAD = False.
HF_HELIUM_REPO = "YOUR_USER/helium-emotion-head"   # change me (or manual copy)
DOWNLOAD_HELIUM_HEAD = True
HELIUM_HEAD_PATH = str(Path(IMTALKER_DIR) / "helium_emotion_head_final.pt")
HELIUM_CONFIG_PATH = str(Path(IMTALKER_DIR) / "config.json")

# Live behaviour (all overridable via env vars in the env cell):
HELIUM_HEAD_DEVICE = "cuda"
HELIUM_TEMPERATURE = 1.0
# Smoother defaults come from config.json next to the head (training
# grid-search). Values > 0 here override config.json; <= 0 = use config.json.
HELIUM_EMA_ALPHA = 0.0
HELIUM_ENTER_MARGIN = 0.0
HELIUM_MIN_DWELL_S = 0.0
# Live-parity defaults (same contract as Phase-B training):
HELIUM_GATE_RMS = 0.006            # smoother only advances while assistant speaks
HELIUM_GRU_RESET_SILENCE_S = 3.0   # GRU context resets at turn boundaries

# Offline replay test on Phase-B test features (needs the emotion_head feature
# dir on THIS pod — set EMO_FEATURES_ROOT = "" to skip):
EMO_FEATURES_ROOT = "/workspace/emotion_head"
RUN_REPLAY_TEST = True
print(f"EMOTION_MODE = {EMOTION_MODE} (AUTO_EMOTION = {AUTO_EMOTION})")
"""

AUTH_CELL = """\
# Hugging Face authentication (needed for the gated nvidia/personaplex-7b-v1
# repo in Step 8; public repos work without it).
# Preferred: `huggingface-cli login` once on the pod, or set HF_TOKEN in the
# RunPod/Jupyter environment. Only as a last resort paste a token into
# HF_TOKEN below — a token committed inside a notebook can leak with any
# screenshot/share of it, so revoke it after the session.
import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from huggingface_hub import get_token
        HF_TOKEN = get_token() or ""
    except Exception:
        HF_TOKEN = ""
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("Hugging Face auth OK (token from env or cached login).")
else:
    print("No HF token found — public repos still work; the gated")
    print("nvidia/personaplex-7b-v1 downloads in Step 8 will 403 until you log in.")
"""

HEAD_FETCH = """\
import shutil
from pathlib import Path
if EMOTION_MODE == "helium":
    if DOWNLOAD_HELIUM_HEAD:
        from huggingface_hub import hf_hub_download
        for remote, local in (("helium_emotion_head_final.pt", HELIUM_HEAD_PATH),
                              ("config.json", HELIUM_CONFIG_PATH)):
            if Path(local).exists():
                print("already present:", local)
                continue
            try:
                p = hf_hub_download(repo_id=HF_HELIUM_REPO, filename=remote)
            except Exception as exc:
                if remote == "config.json":
                    print(f"NOTE: config.json not in {HF_HELIUM_REPO} ({exc!r}) —")
                    print("hardcoded smoother defaults will be used.")
                    continue
                raise
            shutil.copy2(p, local)
            print("downloaded:", local)
    assert Path(HELIUM_HEAD_PATH).exists(), (
        f"helium head not found: {HELIUM_HEAD_PATH} — set HF_HELIUM_REPO / "
        "DOWNLOAD_HELIUM_HEAD or copy the file manually")
    print("helium head ready:", HELIUM_HEAD_PATH,
          f"({Path(HELIUM_HEAD_PATH).stat().st_size / 1e6:.1f} MB)")
else:
    print(f"EMOTION_MODE={EMOTION_MODE}: helium head not needed, skipping.")
"""

# --- helium deploy cell: embeds the deploy modules, runs the patcher, wires sh
HELIUM_SH_BLOCK = """\
# HELIUM_EMOTION_PATCH_V2: agent-emotion head (replaces user-voice SER)
HELIUM_EMOTION="${HELIUM_EMOTION:-0}"
HELIUM_ARGS=()
if [[ "$HELIUM_EMOTION" == "1" ]]; then
  HELIUM_ARGS+=(
    --helium_emotion
    --helium_head_path "${HELIUM_HEAD_PATH:-helium_emotion_head_final.pt}"
    --helium_head_device "${HELIUM_HEAD_DEVICE:-cuda}"
    --helium_temperature "${HELIUM_TEMPERATURE:-1.0}"
    --helium_gate_rms "${HELIUM_GATE_RMS:-0.006}"
    --helium_gru_reset_silence_s "${HELIUM_GRU_RESET_SILENCE_S:-3.0}"
  )
  # values > 0 override the head's config.json smoother (set -e safe: no bare [[ ]] &&)
  if [[ "${HELIUM_EMA_ALPHA:-0}" != "0" ]]; then HELIUM_ARGS+=(--helium_ema_alpha "$HELIUM_EMA_ALPHA"); fi
  if [[ "${HELIUM_ENTER_MARGIN:-0}" != "0" ]]; then HELIUM_ARGS+=(--helium_enter_margin "$HELIUM_ENTER_MARGIN"); fi
  if [[ "${HELIUM_MIN_DWELL_S:-0}" != "0" ]]; then HELIUM_ARGS+=(--helium_min_dwell_s "$HELIUM_MIN_DWELL_S"); fi
fi

"""

BS = chr(92)  # backslash
EXEC_LINE = 'exec python -u "$SERVER_FILE" ' + BS
EMO_LINE = '  "${EMOTION_ARGS[@]}" ' + BS
HELIUM_SH_PATCHES = [
    [EXEC_LINE, HELIUM_SH_BLOCK + EXEC_LINE],
    [EMO_LINE, EMO_LINE + "\n" + '  "${HELIUM_ARGS[@]}" ' + BS],
]

DEPLOY_CELL = (
    """\
import json
import subprocess
if EMOTION_MODE == "helium":
    HELIUM_MODULE = r'''"""
    + HELIUM_MODULE
    + """'''
    INTEGRATE_MODULE = r'''"""
    + INTEGRATE_MODULE
    + """'''
    compile(HELIUM_MODULE, "helium_emotion.py", "exec")
    compile(INTEGRATE_MODULE, "integrate_helium_emotion.py", "exec")
    (Path(IMTALKER_DIR) / "helium_emotion.py").write_text(HELIUM_MODULE, encoding="utf-8")
    (Path(IMTALKER_DIR) / "integrate_helium_emotion.py").write_text(INTEGRATE_MODULE, encoding="utf-8")
    print("[written] helium_emotion.py + integrate_helium_emotion.py ->", IMTALKER_DIR)

    # Server patches: run AFTER Step 12 (the SER patches) — the patcher anchors
    # on exact strings of the SER-patched server (auto-detects SER/PLAIN profile,
    # writes .bak_helium, compile-checks, idempotent).
    r = subprocess.run(
        [VENV_PYTHON, str(Path(IMTALKER_DIR) / "integrate_helium_emotion.py"), SERVER_FILE],
        cwd=IMTALKER_DIR, capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(r.stderr[-2000:])
    assert r.returncode == 0, "helium server patch failed — see output above"
    compile(Path(SERVER_FILE).read_text(encoding="utf-8"), SERVER_FILE, "exec")
    print("[ok] patched server compiles")

    # start_winner_live.sh: HELIUM_ARGS block + exec usage (exact anchors, idempotent)
    HELIUM_SH_PATCHES = json.loads(r'''"""
    + json.dumps(HELIUM_SH_PATCHES, indent=2)
    + """''')
    sh_path = Path(IMTALKER_DIR) / "start_winner_live.sh"
    sh_text = sh_path.read_text(encoding="utf-8")
    if "HELIUM_ARGS" in sh_text:
        print("[skip] start_winner_live.sh already has HELIUM_ARGS")
    else:
        bak = sh_path.with_suffix(sh_path.suffix + ".bak_helium_sh")
        if not bak.exists():
            bak.write_text(sh_text, encoding="utf-8")
        for i, (anchor, repl) in enumerate(HELIUM_SH_PATCHES, 1):
            n = sh_text.count(anchor)
            assert n == 1, f"helium sh anchor #{i} found {n}x (expected 1)"
            sh_text = sh_text.replace(anchor, repl)
        sh_path.write_text(sh_text, encoding="utf-8")
        print("[patched] start_winner_live.sh (backup: .bak_helium_sh)")
else:
    print(f"EMOTION_MODE={EMOTION_MODE}: helium deploy skipped "
          "(the server keeps its SER/fixed behaviour).")
"""
)

HEAD_CHECK_SCRIPT = r'''
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"__IMTALKER_DIR__")
import helium_emotion as hemo

head_path = r"__HEAD__"
payload = torch.load(head_path, map_location="cpu", weights_only=False)
print("head cfg:", payload["cfg"])
print("extra:", payload.get("extra", {}))
dev = "cuda" if torch.cuda.is_available() else "cpu"
runner = hemo.EmotionHeadRunner(head_path, device=dev)
n = sum(p.numel() for p in runner.head.parameters())
print(f"params: {n} ({n/1e6:.2f}M) device={dev}")
cfg_p = Path(head_path).with_name("config.json")
if cfg_p.exists():
    print("config.json:", json.loads(cfg_p.read_text()))
# latency: pure head forward_step (what the live loop pays per 80ms step)
hd = payload["cfg"]
h = torch.randn(1, hd.get("helium_dim", 4096), device=dev)
m = torch.randn(1, hd.get("mimi_dim", 512), device=dev)
p = torch.randn(1, hd.get("pros_dim", 6), device=dev)
hh = None
for _ in range(50):
    _, hh = runner.head.forward_step(h, m, p, hh)
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
N = 1000
for _ in range(N):
    _, hh = runner.head.forward_step(h, m, p, hh)
if dev == "cuda":
    torch.cuda.synchronize()
vram = torch.cuda.memory_allocated() / 1e6 if dev == "cuda" else 0.0
print(f"head forward_step: {(time.perf_counter()-t0)/N*1e6:.1f} us/step "
      f"(budget: 80000 us) vram={vram:.0f} MB")
'''

HEAD_CHECK_CELL = """\
import subprocess
if EMOTION_MODE == "helium":
    HEAD_CHECK = r'''""" + HEAD_CHECK_SCRIPT + """'''
    HEAD_CHECK = HEAD_CHECK.replace("__IMTALKER_DIR__", IMTALKER_DIR).replace("__HEAD__", HELIUM_HEAD_PATH)
    r = subprocess.run([VENV_PYTHON, "-c", HEAD_CHECK], capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr[-2000:])
    assert r.returncode == 0, "head check failed — see output above"
else:
    print(f"EMOTION_MODE={EMOTION_MODE}: head check skipped.")
"""

REPLAY_SCRIPT = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r"__EMO_ROOT__")
sys.path.insert(0, r"__IMTALKER_DIR__")
import eh_common as eh
import helium_emotion as hemo

P = eh.get_paths(Path(r"__EMO_ROOT__"))
assert P.FEAT_B.exists(), f"no phaseB features at {P.FEAT_B} — run notebooks 04/05 here or skip"
recs = [r for r in eh.iter_shard_records(P.FEAT_B, "phaseB") if r["split"] == "test_indomain"]
if not recs:
    recs = list(eh.iter_shard_records(P.FEAT_B, "phaseB"))
recs = recs[:64]
print(f"replaying {len(recs)} Phase-B test clips through the DEPLOY runner + smoother")

dev = "cuda" if torch.cuda.is_available() else "cpu"
runner = hemo.EmotionHeadRunner(r"__HEAD__", device=dev)
cfg_p = Path(r"__HEAD__").with_name("config.json")
sm_cfg = json.loads(cfg_p.read_text()).get("smoother", {}) if cfg_p.exists() else {}
sm = hemo.HysteresisSmoother(
    ema_alpha=float(sm_cfg.get("ema_alpha", 0.4)),
    enter_margin=float(sm_cfg.get("enter_margin", 0.15)),
    min_dwell_s=float(sm_cfg.get("min_dwell_s", 1.5)))
names = ["neutral", "angry", "happy", "sad"]
conf = np.zeros((4, 4), dtype=int)
sw = []
argmax_ok = argmax_n = 0
for r in recs:
    sm.reset()
    runner.reset()
    label = int(r["label"])
    ids = []
    hh = None
    T = r["hidden"].shape[0]
    for t in range(T):
        if not bool(r["agent_mask"][t]):
            continue
        h = r["hidden"][t].float().unsqueeze(0).to(dev)
        m = r["mimi_lat"][t].float().unsqueeze(0).to(dev)
        p = r["pros"][t].float().unsqueeze(0).to(dev)
        logits, hh = runner.head.forward_step(h, m, p, hh)
        probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        argmax_ok += int(int(probs.argmax()) == label)
        argmax_n += 1
        sm.update(probs)
        ids.append(sm.current)
    if ids:
        pred = max(set(ids), key=ids.count)
        conf[label, pred] += 1
    sw.append(sm.switches_per_min())
print(f"frame argmax acc: {argmax_ok}/{argmax_n} = {argmax_ok/max(1,argmax_n):.3f}")
print("clip-level confusion (rows=true, cols=smoothed-majority):")
print("         " + " ".join(f"{n:>8}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>8} " + " ".join(f"{int(conf[i, j]):8d}" for j in range(4)))
recall = [conf[i, i] / max(1, int(conf[i].sum())) for i in range(4)]
print("per-class recall:", {n: round(float(recall[i]), 3) for i, n in enumerate(names)})
print(f"switches/min: mean={np.mean(sw):.2f} p95={np.percentile(sw, 95):.2f}")
print("(expect per-class recall well above 0.25 chance; if angry/happy recall is low,")
print(" tune the smoother via HELIUM_EMA_ALPHA / HELIUM_ENTER_MARGIN / HELIUM_MIN_DWELL_S)")
'''

REPLAY_CELL = """\
import subprocess
if EMOTION_MODE == "helium" and RUN_REPLAY_TEST:
    if EMO_FEATURES_ROOT and Path(EMO_FEATURES_ROOT).exists():
        REPLAY = r'''""" + REPLAY_SCRIPT + """'''
        REPLAY = (REPLAY.replace("__EMO_ROOT__", EMO_FEATURES_ROOT)
                        .replace("__IMTALKER_DIR__", IMTALKER_DIR)
                        .replace("__HEAD__", HELIUM_HEAD_PATH))
        r = subprocess.run([VENV_PYTHON, "-c", REPLAY], capture_output=True, text=True)
        print(r.stdout[-5000:])
        print(r.stderr[-2000:])
        assert r.returncode == 0, "replay test failed — see output above"
    else:
        print("EMO_FEATURES_ROOT not found on this pod — replay skipped.")
        print("(train notebooks 02-05 there, or set RUN_REPLAY_TEST = False)")
else:
    print(f"replay skipped (EMOTION_MODE={EMOTION_MODE}, RUN_REPLAY_TEST={RUN_REPLAY_TEST}).")
"""

ENV_CELL = """\
env_overrides = {
    "HOST": HOST,
    "PORT": str(PORT),
    "VENV_DIR": VENV_DIR,
    "SPEECH2AVATAR_ROOT": PROJECT_ROOT,
    # Emotion conditioning — read by the patched start_winner_live.sh, which
    # turns them into --emotion_* server args (see Step 12).
    "EMOTION_ID": str(EMOTION_ID),
    "EMOTION_LORA_PATH": EMOTION_ADAPTER_PATH,
    "EMOTION_LORA_RANK": str(EMOTION_LORA_RANK),
    "EMOTION_LORA_ALPHA": str(EMOTION_LORA_ALPHA),
    # Emotion-only CFG scale (compositional guidance; audio CFG stays at A_CFG_SCALE).
    "EMOTION_CFG_SCALE": str(EMOTION_CFG_SCALE),
    "EMOTION_CFG_BOOST_HAPPY": str(EMOTION_CFG_BOOST_HAPPY),
    # Silence-aware emotion CFG (SILENCE_EMOTION_PATCH_V1): rebalanced CFG
    # scales while the avatar is not speaking (see Parameters cell).
    "A_CFG_SCALE_SILENCE": str(A_CFG_SCALE_SILENCE),
    "EMOTION_CFG_SCALE_SILENCE": str(EMOTION_CFG_SCALE_SILENCE),
    "GATE_CFG_RAMP_CHUNKS": str(GATE_CFG_RAMP_CHUNKS),
    "EMOTION_SWITCH_RAMP_CHUNKS": str(EMOTION_SWITCH_RAMP_CHUNKS),
    # AUTO_EMOTION (V3) - voice-based automatic emotion detection
    # (only active when EMOTION_MODE = "ser").
    "AUTO_EMOTION": "1" if EMOTION_MODE == "ser" else "0",
    "SER_MODEL": SER_MODEL,
    "SER_DEVICE": SER_DEVICE,
    "SER_CONFIDENCE_THRESHOLD": str(SER_CONFIDENCE_THRESHOLD),
    "VAD_RMS_THRESHOLD": str(VAD_RMS_THRESHOLD),
    "VAD_SILENCE_SEC": str(VAD_SILENCE_SEC),
    "VAD_MIN_SPEECH_SEC": str(VAD_MIN_SPEECH_SEC),
    "VAD_MAX_UTTERANCE_SEC": str(VAD_MAX_UTTERANCE_SEC),
    # HELIUM agent-emotion (this notebook) - the avatar's expression follows the
    # ASSISTANT's generated speech via the trained helium-hidden GRU head
    # (only active when EMOTION_MODE = "helium").
    "HELIUM_EMOTION": "1" if EMOTION_MODE == "helium" else "0",
    "HELIUM_HEAD_PATH": HELIUM_HEAD_PATH,
    "HELIUM_HEAD_DEVICE": HELIUM_HEAD_DEVICE,
    "HELIUM_TEMPERATURE": str(HELIUM_TEMPERATURE),
    "HELIUM_GATE_RMS": str(HELIUM_GATE_RMS),
    "HELIUM_GRU_RESET_SILENCE_S": str(HELIUM_GRU_RESET_SILENCE_S),
    # Renderer sub-batch: frames decoded per render call. The renderer's attention
    # needs ~0.5 GiB/frame; on a loaded 32 GB 5090 only ~4.5 GiB is free after the
    # models, so a whole 24-frame chunk (12 GiB) OOMs. 8 is the safe maximum.
    "RENDER_SUB_BATCH": "8",
}
if REF_PATH:
    env_overrides["REF_PATH"] = REF_PATH
if A_CFG_SCALE:
    env_overrides["A_CFG_SCALE"] = str(A_CFG_SCALE)
if NFE:
    env_overrides["NFE"] = str(NFE)
if PROMPT_FILE:
    env_overrides["PROMPT_FILE"] = PROMPT_FILE
if VOICE_PROMPT:
    env_overrides["VOICE_PROMPT"] = VOICE_PROMPT
if VOICE_PROMPT_DIR:
    env_overrides["VOICE_PROMPT_DIR"] = VOICE_PROMPT_DIR
if LORA_GENERATOR_PATH:
    env_overrides["LORA_GENERATOR_PATH"] = LORA_GENERATOR_PATH
if DISABLE_LORA:
    env_overrides["DISABLE_LORA"] = str(DISABLE_LORA)
if JITTER_BUFFER_SEC:
    env_overrides["JITTER_BUFFER_SEC"] = str(JITTER_BUFFER_SEC)
# Smoother overrides (only passed when > 0; otherwise config.json next to the
# head supplies the training grid-search values).
if HELIUM_EMA_ALPHA > 0:
    env_overrides["HELIUM_EMA_ALPHA"] = str(HELIUM_EMA_ALPHA)
if HELIUM_ENTER_MARGIN > 0:
    env_overrides["HELIUM_ENTER_MARGIN"] = str(HELIUM_ENTER_MARGIN)
if HELIUM_MIN_DWELL_S > 0:
    env_overrides["HELIUM_MIN_DWELL_S"] = str(HELIUM_MIN_DWELL_S)


print("Resolved environment overrides for run_live.sh:")
for k, v in env_overrides.items():
    print(f"  {k}={v}")
"""

# health cell (46) additions — applied via string surgery on the original cell
HEALTH_MARKER_ANCHOR = '    HEALTHY_MARKERS["style LoRA merged into base"] = "style LoRA layers"\n'
HEALTH_MARKER_ADD = """\
if EMOTION_MODE == "helium":
    HEALTHY_MARKERS["helium emotion manager enabled"] = "[HELIUM-EMOTION] enabled:"
"""
HEALTH_HINTS_ANCHOR = "}\n\n\ndef wait_for_healthy"
HEALTH_HINTS_ADD = """\
}

if EMOTION_MODE == "helium":
    ERROR_HINTS.setdefault(
        "[HELIUM-EMOTION] disabled",
        "helium_emotion.py did not deploy (helium deploy step) — check its output above")
    ERROR_HINTS.setdefault(
        "helium head not found",
        "the head download failed — check HF_HELIUM_REPO or copy the head manually")
    ERROR_HINTS.setdefault(
        "[patch] ABORT",
        "helium patch anchors mismatched — re-run Step 12, then the helium deploy step")


def wait_for_healthy"""
HEALTH_TAIL_ADD = """
# Helium sanity: warn (don't fail) if the head manager didn't come up.
if EMOTION_MODE == "helium" and os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r", errors="ignore") as f:
        _log = f.read()
    if "[HELIUM-EMOTION] enabled:" in _log:
        print("[ok] helium agent-emotion head is active (expression follows the assistant)")
    else:
        print("[warn] EMOTION_MODE=helium but no '[HELIUM-EMOTION] enabled:' line in the log —")
        print("       check HELIUM_HEAD_PATH and the helium deploy step output.")
"""

# success-banner cell (48) surgery: EMOTION_MODE-aware banner
READY_ANCHOR = 'print(f"  Emotion: {EMOTION} (id={EMOTION_ID}) — fixed for this entire session")\n'
READY_REPL = """\
if EMOTION_MODE == "helium":
    print(f"  Emotion mode: HELIUM agent-emotion head ({HELIUM_HEAD_PATH})")
    print(f"    expression follows the ASSISTANT's speech; switch CSV: {IMTALKER_DIR}/emotion_logs/")
elif EMOTION_MODE == "ser":
    print(f"  Emotion mode: SER on YOUR voice (fallback {EMOTION}, id={EMOTION_ID})")
else:
    print(f"  Emotion: {EMOTION} (id={EMOTION_ID}) — fixed for this entire session")
"""

# explainer cell (53) replacement
EXPLAIN_CELL = """\
# How emotion control works (EMOTION_MODE)
# ----------------------------------------
# "helium" (default): the avatar's expression follows the ASSISTANT's generated
# speech. Every 80ms the server's reply pipeline feeds (helium hidden state,
# mimi reply latents, reply prosody) into the 2.03M-param GRU head; the EMA +
# hysteresis smoother publishes an emotion id that overrides _emotion_id in the
# frame generator. The smoother only advances while the assistant is actually
# speaking (reply-RMS gate), and the GRU context resets after 3s of assistant
# silence — the same contract the head was trained with (Phase B). Watch the
# log for "[HELIUM-EMOTION] neutral -> angry (p=0.87)" style lines; the CSV
# timeline lands in IMTalker/emotion_logs/.
#
# "ser": the ORIGINAL behaviour — AUTO_EMOTION listens to YOUR mic, classifies
# each utterance with emotion2vec, and switches the avatar's expression to
# mirror you (~1s after the utterance ends).
#
# "fixed": EMOTION stays fixed for the whole session.
#
# In all modes the emotion LoRA + silence-aware CFG from Step 12 render the
# actual facial expression; the mode only decides WHO picks the emotion id.
# Tuning (env vars, then re-run the env cell + restart): HELIUM_EMA_ALPHA /
# HELIUM_ENTER_MARGIN / HELIUM_MIN_DWELL_S override config.json next to the
# head; HELIUM_GATE_RMS / HELIUM_GRU_RESET_SILENCE_S change the live gating.
"""

MONITOR_CELL = """\
import glob
import subprocess
if EMOTION_MODE == "helium":
    print("--- server log [HELIUM-EMOTION] ---")
    logs = sorted(glob.glob(str(Path(IMTALKER_DIR) / "logs" / "server_*.log")),
                  key=os.path.getmtime)
    if logs:
        out = subprocess.run(["grep", "-h", "HELIUM-EMOTION"] + logs[-2:],
                             capture_output=True, text=True).stdout
        print("\\n".join(out.splitlines()[-20:]) or "(no [HELIUM-EMOTION] lines yet)")
    else:
        print("(no server log yet)")
    print("--- emotion timeline CSV (latest) ---")
    csvs = sorted(glob.glob(str(Path(IMTALKER_DIR) / "emotion_logs" / "helium_emotion_*.csv")),
                  key=os.path.getmtime)
    if csvs:
        print(csvs[-1])
        print("\\n".join(Path(csvs[-1]).read_text().splitlines()[-15:]))
    else:
        print("(no CSV yet — the manager writes one row per switch + a 2s heartbeat)")
else:
    print(f"EMOTION_MODE={EMOTION_MODE} — helium monitor not applicable.")
"""

# ---------------------------------------------------------------------------
# Assemble the new notebook
# ---------------------------------------------------------------------------
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def csrc(c):
    return "".join(c["source"])


def find(pred, desc):
    hits = [i for i, c in enumerate(cells) if pred(c)]
    if len(hits) != 1:
        sys.exit(f"locate {desc}: {len(hits)} hits (expected 1)")
    return hits[0]


i_params = find(lambda c: 'IMTALKER_DIR = f"{PROJECT_ROOT}/IMTalker"' in csrc(c), "params code cell")
i_auth = find(lambda c: c.get("cell_type") == "code" and "HF_TOKEN" in csrc(c)
              and "getpass" in csrc(c), "HF auth cell")
i_assets = find(lambda c: "REQUIRED_ASSETS" in csrc(c) and "tarfile" in csrc(c), "assets cell")
i_patch = find(lambda c: c.get("id") == "emotion-stream-33", "Step-12 patch cell")
i_validate = find(lambda c: "PATCH_CHECKS = [" in csrc(c) and "patch_ok" in csrc(c), "validation cell")
i_env = find(lambda c: "env_overrides = {" in csrc(c), "env cell")
i_health = find(lambda c: "HEALTHY_MARKERS = {" in csrc(c), "health cell")
i_ready = find(lambda c: "SUCCESS: PersonaPlex + IMTalker emotion live server" in csrc(c), "banner cell")
i_monitor = find(lambda c: "# Tail the live server log" in csrc(c), "log tail cell")
i_explain = find(lambda c: "How emotion control works" in csrc(c), "explainer cell")

# validation cell: keep everything, add conditional helium PATCH_CHECKS entries
HELIUM_CHECKS_SNIPPET = """\

if EMOTION_MODE == "helium":
    PATCH_CHECKS += [
        (SERVER_FILE, ["HELIUM_EMOTION_PATCH_V2", "--helium_emotion",
                       "init_helium_emotion_manager(args)", "_hem.feed_step(",
                       "_emotion_id = _hem.current_emotion_id"]),
        (START_WINNER_SH, ["HELIUM_ARGS", "--helium_emotion", '"${HELIUM_ARGS[@]}"']),
        (Path(IMTALKER_DIR) / "helium_emotion.py", ["class HeliumEmotionManager"]),
    ]
"""
orig_validate = csrc(cells[i_validate])
_validate_anchor = "]\npatch_ok = True"
assert _validate_anchor in orig_validate, "validation cell PATCH_CHECKS anchor missing"
VALIDATE_CELL = orig_validate.replace(
    _validate_anchor, "]" + HELIUM_CHECKS_SNIPPET + "\npatch_ok = True", 1)

orig_health = csrc(cells[i_health])
assert HEALTH_MARKER_ANCHOR in orig_health, "health marker anchor missing"
assert HEALTH_HINTS_ANCHOR in orig_health, "health hints anchor missing"
HEALTH_CELL = (orig_health
               .replace(HEALTH_MARKER_ANCHOR, HEALTH_MARKER_ANCHOR + HEALTH_MARKER_ADD, 1)
               .replace(HEALTH_HINTS_ANCHOR, HEALTH_HINTS_ADD, 1)
               + HEALTH_TAIL_ADD)

orig_ready = csrc(cells[i_ready])
assert READY_ANCHOR in orig_ready, "banner anchor missing"
READY_CELL = orig_ready.replace(READY_ANCHOR, READY_REPL, 1)

replacements = {
    0: md("title", TITLE_MD),
    i_auth: code("hf-auth", AUTH_CELL),
    i_validate: code("validate-patches", VALIDATE_CELL),
    i_env: code("env-overrides", ENV_CELL),
    i_health: code("health-wait", HEALTH_CELL),
    i_ready: code("success-banner", READY_CELL),
    i_explain: code("emotion-mode-explainer", EXPLAIN_CELL),
}
insertions = {
    i_params: [code("helium-params", HELIUM_PARAMS)],
    i_assets: [code("helium-head-fetch", HEAD_FETCH)],
    i_patch: [code("helium-deploy", DEPLOY_CELL)],
    i_validate: [code("helium-head-check", HEAD_CHECK_CELL),
                 code("helium-replay", REPLAY_CELL)],
    i_monitor: [code("helium-monitor", MONITOR_CELL)],
}

new_cells = []
for i, c in enumerate(cells):
    new_cells.append(replacements.get(i, c))
    new_cells.extend(insertions.get(i, []))

# AST-validate every code cell (skip shell cells)
for c in new_cells:
    if c["cell_type"] != "code":
        continue
    s = csrc(c)
    if s.lstrip().startswith("!"):
        continue
    try:
        ast.parse(s)
    except SyntaxError as exc:
        sys.exit(f"cell id={c.get('id')} does not compile: {exc}")

nb["cells"] = new_cells
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT_NB.name}: {len(new_cells)} cells ({len(cells)} original)")

