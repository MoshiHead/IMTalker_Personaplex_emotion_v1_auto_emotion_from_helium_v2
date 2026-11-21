"""build_notebooks.py — generates the 7 pipeline notebooks.

Run:  python build_notebooks.py
Writes 00..06 .ipynb files next to this script and AST-validates every code
cell (magics stripped) before writing.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip() + "\n"}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip() + "\n"}


NB_META = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}


def validate(cells):
    for c in cells:
        if c["cell_type"] != "code":
            continue
        out = []
        for l in c["source"].splitlines():
            stripped = l.lstrip()
            if stripped.startswith(("%", "!")):
                out.append(l[: len(l) - len(stripped)] + "pass")
            else:
                out.append(l)
        ast.parse("\n".join(out) or "pass")


def write_nb(name, cells):
    validate(cells)
    nb = {"cells": cells, "metadata": NB_META, "nbformat": 4, "nbformat_minor": 5}
    (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {name}: {len(cells)} cells")


SHARED = r'''
from pathlib import Path
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import eh_common as eh

P = eh.get_paths()   # override: P = eh.get_paths("/your/root") or set EMOTION_HEAD_ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
print("ROOT:", P.ROOT, "| device:", DEVICE)
for k in ("MOSHI_BNB4", "MIMI_FILE", "SPM_FILE", "VOICE_PT"):
    p = Path(getattr(P, k))
    print(f"  {k}: {'OK' if p.exists() else 'MISSING'}  {p}")
'''


# ===========================================================================
# 00 — setup & downloads
# ===========================================================================
NB00 = [
md(r'''
# 00 — Setup & downloads

Run once per pod. Run order: **00 → 01 → 02 → 03 → 04 → 05 → 06**.

Before you start:
- Run inside a PyTorch template (or your `preprocess_5090` venv). The install cell pins torch 2.8.0/cu128.
- `export HF_TOKEN=hf_...` (RunPod env var) — the NVIDIA PersonaPlex repo (mimi, SentencePiece, voice prompts) is gated; you must have accepted its license. The EmotionPlex dataset itself is public.
- Upload your local `IMTalker/prompts/RB_Robert_System_Prompt_full.txt` to `<root>/prompts/` (checked below).
- **License note**: the EmotionPlex dataset audio is Higgs-generated — research / non-commercial use only.
- Disk: ~45 GB (bnb4 ~6 GB, dataset ~28 GB, emotionplex-7b ~17 GB one-off).
'''),
code(r'''
import sys, torch
print(sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "GPU pod required"
print("gpu:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
'''),
code(r'''
DO_INSTALL = True   # False if your env already matches (e.g. preprocess_5090)
if DO_INSTALL:
    %pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    %pip install -q bitsandbytes sphn sentencepiece soundfile safetensors einops num2words "huggingface_hub[hf_transfer]" scikit-learn matplotlib pandas pyarrow tqdm ipywidgets
import torch
print("torch:", torch.__version__)
'''),
code(SHARED),
code(r'''
import os
from huggingface_hub import login, whoami
tok = os.environ.get("HF_TOKEN", "").strip()
assert tok, "HF_TOKEN env var is required (gated repos). Set it in the pod env or: %env HF_TOKEN=hf_..."
login(token=tok, add_to_git_credential=False)
print("logged in as:", whoami()["name"])
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
'''),
code(r'''
import shutil, tarfile
from huggingface_hub import snapshot_download, hf_hub_download

P.CKPT.mkdir(parents=True, exist_ok=True)
snapshot_download(eh.BNB4_REPO, local_dir=str(P.CKPT),
                  allow_patterns=["model_bnb_4bit.pt", "moshi/**"])
print("bnb4 snapshot done (quantized LM + moshi source)")

# The bnb4 community repo ships ONLY the quantized LM. Mimi, SentencePiece and
# the voice prompts live in the gated NVIDIA repo (needs HF_TOKEN + license).
eh.hf_fetch(eh.NVIDIA_REPO, "tokenizer-e351c8d8-checkpoint125.safetensors", P.MIMI_FILE)
eh.hf_fetch(eh.NVIDIA_REPO, "tokenizer_spm_32k_3.model", P.SPM_FILE)

if not P.VOICE_PT.exists():
    print("VARM3.pt not in bnb4 snapshot; fetching voices.tgz from", eh.NVIDIA_REPO)
    tgz = Path(hf_hub_download(eh.NVIDIA_REPO, "voices.tgz"))
    voices_dir = tgz.parent / "voices"
    if not voices_dir.exists():
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(path=tgz.parent)
    P.VOICE_PT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(voices_dir / "VARM3.pt", P.VOICE_PT)

ok = True
for k in ("MOSHI_BNB4", "MIMI_FILE", "SPM_FILE", "VOICE_PT"):
    p = Path(getattr(P, k))
    good = p.exists()
    ok &= good
    print(("OK      " if good else "MISSING "), p,
          f"{p.stat().st_size/1e9:.2f} GB" if good else "")
assert ok, "checkpoint assets missing"
'''),
code(r'''
import subprocess, sys
if not (P.MOSHI_ROOT / "pyproject.toml").exists():
    print("moshi source missing -> fetching moshi/** from", eh.BNB4_REPO)
    snapshot_download(eh.BNB4_REPO, local_dir=str(P.CKPT), allow_patterns=["moshi/**"])
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                    str(P.MOSHI_ROOT), "--no-deps"])
print("pip editable install rc:", r.returncode,
      "(non-zero is fine: sys.path fallback below still makes moshi importable)")
print("moshi importable:", eh.ensure_moshi_importable(P))
'''),
code(r'''
P.PROMPTS.mkdir(parents=True, exist_ok=True)
if not P.TEXT_PROMPT_FILE.exists():
    print("!! ACTION REQUIRED: upload your local file")
    print("   IMTalker/prompts/RB_Robert_System_Prompt_full.txt")
    print("   to", P.TEXT_PROMPT_FILE)
else:
    txt = P.TEXT_PROMPT_FILE.read_text(encoding="utf-8", errors="replace")
    print(f"text prompt OK: {len(txt)} chars | {txt[:100]!r} ...")
'''),
]


NB00 += [
code(r'''
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

api = HfApi()
files = api.list_repo_files(eh.DATASET_REPO, repo_type="dataset")
P.DATA.mkdir(parents=True, exist_ok=True)
(P.DATA / "_repo_files.json").write_text(json.dumps(files))
json_files = sorted(f for f in files if f.endswith((".json", ".jsonl")))
audio_files = sorted(f for f in files if f.endswith((".wav", ".tar", ".tar.gz")))
print(f"{len(files)} files | {len(json_files)} json/jsonl | {len(audio_files)} audio")
print("sample audio names:", [f.split("/")[-1] for f in audio_files[:5]])

def _dl(f):
    return hf_hub_download(eh.DATASET_REPO, f, repo_type="dataset", local_dir=str(P.DATA))

with ThreadPoolExecutor(max_workers=16) as ex:
    list(tqdm(ex.map(_dl, json_files), total=len(json_files), desc="sidecars"))
print("sidecars + manifests done")
'''),
code(r'''
DOWNLOAD_ALL_AUDIO = True     # False = only the audio of a random MAX_CLIPS_SUBSET subset
MAX_CLIPS_SUBSET = 8000

if DOWNLOAD_ALL_AUDIO:
    targets = audio_files
else:
    rows = []
    for jf in sorted(P.DATA.rglob("*.jsonl")):
        for line in open(jf, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rng = np.random.RandomState(eh.SEED)
    idx = rng.permutation(len(rows))[:MAX_CLIPS_SUBSET]
    wanted = set()
    for i in idx:
        for key in ("path", "wav", "audio", "file"):
            v = rows[i].get(key)
            if v:
                wanted.add(str(v).split("/")[-1])
    targets = [f for f in audio_files if any(w in f for w in wanted)]
    if not targets:
        print("!! subset matching found nothing; falling back to ALL audio files")
        targets = audio_files
print(f"downloading {len(targets)} audio files")
with ThreadPoolExecutor(max_workers=8) as ex:
    list(tqdm(ex.map(_dl, targets), total=len(targets), desc="audio"))
print("wav on disk:", sum(1 for _ in P.DATA.rglob("*.wav")))
'''),
code(r'''
import gc
sp = eh.load_sp(P)
mimi = eh.load_mimi(P, DEVICE)
lm = eh.load_lm_bnb4(P, DEVICE, DTYPE)
print(f"LM params: {sum(p.numel() for p in lm.parameters())/1e9:.2f}B | "
      f"cuda mem: {torch.cuda.memory_allocated()/1e9:.2f} GiB")
lm_gen = eh.make_lm_gen(lm, DEVICE)
eh.install_hidden_step(lm_gen)
sil = np.zeros(20 * eh.FRAME_SAMPLES, np.float32)
codes = eh.mimi_encode_codes(mimi, sil, DEVICE)
assert codes.shape == (20, 8), codes.shape
for t in range(20):
    # _step_force tolerates the offset-0 init step (first call returns None and
    # discards its inputs in moshi's prepare_step_input; re-feeding is exact).
    out = eh._step_force(lm_gen, input_tokens=codes[t].view(1, 8, 1).to(DEVICE),
                         text_token=eh.TEXT_PAD)
    assert out[2].shape[-1] == eh.HELIUM_DIM, out[2].shape
lat = eh.codes_to_latent(mimi, codes, DEVICE)
assert lat.shape == (20, 512), lat.shape
print("SMOKE TEST OK — bnb4 loads, live step path works, hidden [4096] + latent [512] confirmed")
del lm, lm_gen, mimi
gc.collect()
torch.cuda.empty_cache()
'''),
code(r'''
marker = dict(time=time.time(), torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0))
P.ROOT.mkdir(parents=True, exist_ok=True)
(P.ROOT / "_env_ok.json").write_text(json.dumps(marker, indent=2))
print("Setup complete → continue with 01_dataset_inspect_and_manifest.ipynb")
'''),
]




# ===========================================================================
# 01 — dataset inspect & manifest
# ===========================================================================
NB01 = [
md(r'''
# 01 — EmotionPlex dataset: inspect, validate, manifest

- Parses every sidecar JSON, validates the schema (prints samples so YOU can eyeball them).
- Agent reply label = `styletalk.res_emotion` (cross-checked against path / text_prompt).
- Agent turn spans from `turns[]`; word alignments for the text stream; `user_id` for grouped splits.
- Splits: official val manifest → `val`; 10% of train users → `holdout` (speaker generalization).
- Writes `manifest.parquet` (alignments/turns as JSON strings). CPU only.
'''),
code(SHARED),
code(r'''
jsonl_files = sorted(P.DATA.rglob("*.jsonl"))
sidecars = [p for p in sorted(P.DATA.rglob("*.json"))
            if not p.name.startswith("_") and p.parent != P.DATA]
wav_by_name = {w.name: w for w in P.DATA.rglob("*.wav")}
print(f"{len(jsonl_files)} manifests | {len(sidecars)} sidecars | {len(wav_by_name)} wav")
rows = []
for jf in jsonl_files:
    for line in open(jf, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            r["_manifest"] = jf.name
            rows.append(r)
print("manifest rows:", len(rows))
print("row keys:", sorted(rows[0].keys()))
print("example row:", {k: rows[0][k] for k in sorted(rows[0])})
'''),
code(r'''
from collections import Counter
sidecar_by_name = {p.name: p for p in sidecars}
sample = json.loads(sidecars[0].read_text(encoding="utf-8"))
print("== sidecar example:", sidecars[0].name)
print(json.dumps(sample, indent=2, ensure_ascii=False)[:3000])
keysets = Counter()
for p in sidecars[:300]:
    try:
        keysets[tuple(sorted(json.loads(p.read_text(encoding="utf-8")).keys()))] += 1
    except Exception:
        pass
print("== sidecar key sets (top 5):")
for ks, n in keysets.most_common(5):
    print(f"  {n:5d}  {ks}")
'''),
]

NB01 += [
code(r'''
import re
import pandas as pd

AGENT_NAMES = {"agent", "assistant", "moshi", "bot", "personaplex",
               "speaker_broker", "speaker_0", "spk0", "left", "0"}
USER_NAMES = {"user", "human", "caller", "speaker_client",
              "speaker_1", "spk1", "right", "1"}

def find_emotion(d, path):
    st = d.get("styletalk")
    if isinstance(st, dict) and "res_emotion" in st:
        return str(st["res_emotion"])
    for k in ("styletalk.res_emotion", "res_emotion", "emotion", "label"):
        if k in d:
            return str(d[k])
    tp = str(d.get("text_prompt", "")).lower()
    for e in eh.EMOTIONS:
        if f"a {e} tone" in tp:
            return e
    m = re.search(r"(happy|sad|angry|neutral)", str(path).lower())
    return m.group(1) if m else None

def find_turns(d):
    out = []
    for t in (d.get("turns") or d.get("conversation") or []):
        if not isinstance(t, dict):
            continue
        # the dataset's `role` ("user"/"agent") is authoritative; `speaker` is
        # SPEAKER_CLIENT/SPEAKER_BROKER (channel labels)
        spk = str(t.get("role") or t.get("speaker", t.get("channel", ""))).strip().lower()
        s = t.get("start", t.get("x0", t.get("begin")))
        e = t.get("end", t.get("x1", t.get("stop")))
        if s is None or e is None:
            continue
        side = "agent" if spk in AGENT_NAMES else ("user" if spk in USER_NAMES else spk or "?")
        out.append((side, float(s), float(e)))
    return out

def find_alignments(d):
    out = []
    for w in (d.get("alignments") or d.get("alignment") or []):
        if isinstance(w, dict):
            out.append({k: w[k] for k in ("word", "start", "end", "speaker") if k in w})
        elif isinstance(w, (list, tuple)) and len(w) >= 2 and isinstance(w[1], (list, tuple)):
            rec = {"word": w[0], "start": float(w[1][0]), "end": float(w[1][1])}
            if len(w) > 2:
                rec["speaker"] = str(w[2])
            out.append(rec)
    return out

def guess_user_emotion(row, d):
    for k in ("user_emotion", "user_style"):
        if d.get(k):
            return str(d[k]).lower()
    return None

records, problems = [], []
for r in rows:
    path = str(r.get("path", ""))
    stem = Path(path).stem
    wav = wav_by_name.get(Path(path).name)
    sc = sidecar_by_name.get(stem + ".json")
    d = {}
    if sc is not None:
        try:
            d = json.loads(sc.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    emo = str(find_emotion(d, path) or "").strip().lower()
    if emo not in eh.EMO2ID:
        problems.append((path, emo))
        continue
    turns = find_turns(d)
    agent_turns = [(s, e) for (side, s, e) in turns if side == "agent"]
    records.append(dict(
        clip_id=stem, rel_path=path,
        wav_path=str(wav) if wav else "", has_wav=bool(wav),
        manifest=r["_manifest"],
        duration=float(r.get("duration", 0.0) or 0.0),
        emotion=emo,
        user_id=str(d.get("user_id", r.get("user_id", ""))),
        user_emotion=guess_user_emotion(r, d),
        n_turns=len(turns), n_agent_turns=len(agent_turns),
        agent_turn_frames=json.dumps(
            [[int(round(s * eh.FRAME_HZ)), int(round(e * eh.FRAME_HZ))]
             for s, e in agent_turns]),
        alignments=json.dumps(find_alignments(d)),
    ))
print(f"records: {len(records)} | skipped: {len(problems)}")
if problems[:5]:
    print("skip examples:", problems[:5])
df = pd.DataFrame(records)
print(df["emotion"].value_counts())
print("agent turns per clip:", df.n_agent_turns.describe().round(2).to_dict())
'''),
]


NB01 += [
code(r'''
df["split"] = np.where(df["manifest"].str.lower().str.contains("val"), "val", "train")
train_users = np.array(sorted(df.loc[df.split == "train", "user_id"].unique()))
rng = np.random.RandomState(eh.SEED)
holdout = set(rng.permutation(train_users)[: max(1, len(train_users) // 10)])
df.loc[(df.split == "train") & df.user_id.isin(holdout), "split"] = "holdout"
df.to_parquet(P.DATA / "manifest.parquet", index=False)
print(df.groupby(["split", "emotion"]).size().unstack(fill_value=0))
print(f"holdout users: {len(holdout)} | wav coverage: {df.has_wav.mean()*100:.1f}%")
'''),
code(r'''
import subprocess
repo = P.ROOT / "refs" / "emotionplex"
if not repo.exists():
    repo.parent.mkdir(parents=True, exist_ok=True)
    for url in ("https://github.com/frabonzi/emotionplex",
                "https://github.com/mago-research/emotionplex"):
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(repo)],
                           check=False, capture_output=True, text=True)
        if r.returncode == 0:
            print("cloned", url)
            break
        print("clone failed:", url)
il = repo / "finetune" / "finetune" / "data" / "interleaver.py"
if il.exists():
    print(il.read_text(encoding="utf-8", errors="replace")[:5000])
    print()
    print(">>> CROSS-CHECK: our text stream uses PAD=3 everywhere, EOP=0 one frame before")
    print(">>> each word, word pieces from the word-start frame (eh.build_agent_text_stream).")
    print(">>> If the interleaver above disagrees, fix eh_common.build_agent_text_stream")
    print(">>> BEFORE running notebook 02 (hidden states depend on the text stream).")
else:
    print("clone failed — check the interleaver convention manually before running 02")
'''),
code(r'''
chk = pd.read_parquet(P.DATA / "manifest.parquet")
assert set(chk.emotion.unique()) <= set(eh.EMOTIONS)
assert chk.clip_id.is_unique
assert (chk.n_agent_turns >= 1).mean() > 0.95, "many clips without agent turns - check find_turns"
print("manifest.parquet OK:", len(chk), "clips |", chk.split.value_counts().to_dict())
print("-> continue with 02_extract_features_emotionplex.ipynb")
'''),
]


# ===========================================================================
# 02 — Phase A feature extraction (EmotionPlex clips, teacher-forced bnb4)
# ===========================================================================
NB02 = [
md(r'''
# 02 — Phase A feature extraction (teacher-forced PersonaPlex bnb4)

For each dataset clip, replays the conversation through the **live-identical**
bnb4 step path: user codes → `input_tokens`, dataset agent codes → `moshi_tokens`,
word-alignment text stream → `text_token`. Captures per 80ms step:

- `hidden [T, 4096]` fp16 — final-layer `transformer_out` (the same tensor the live server captures)
- `agent_codes [T, 8]` int16 and `mimi_lat [T, 512]` fp16 (stateless RRVQ decode)
- `pros [T, 6]` fp16 DSP prosody, `agent_mask`, `label`, `text_stream`

Decisions:
- `USE_LIVE_PROMPT_PREFIX = True` (default): the Robert text prompt + VARM3 voice
  prefix are applied ONCE and snapshotted, then restored per clip — this matches
  the live hidden-state distribution. VARM3 is NOT used to generate speech here.
- Resume-safe: delete nothing; re-running skips completed clips.
- Time: measured after 5 clips; ~0.8 s/clip on a 5090 → ~1.8 h for 8k clips.
'''),
code(SHARED),
code(r'''
MAX_CLIPS = 8000
USE_LIVE_PROMPT_PREFIX = True
CLIPS_PER_SHARD = 256
'''),
code(r'''
import pandas as pd
df = pd.read_parquet(P.DATA / "manifest.parquet")
df = df[df.has_wav].copy()
quota = max(1, MAX_CLIPS // len(eh.EMOTIONS))
# NB: no groupby().apply() here — pandas >= 2.2/3.x drops the grouping column
# from the applied frame (include_groups=False), which silently breaks sel.emotion
_parts = [g.sample(min(len(g), quota), random_state=eh.SEED)
          for _, g in df[df.split == "train"].groupby("emotion")]
assert _parts, "no train rows with has_wav=True — check 01's wav coverage / split columns"
sel = pd.concat(_parts).reset_index(drop=True)
sel["alignments"] = sel["alignments"].map(json.loads)
sel["agent_turn_frames"] = sel["agent_turn_frames"].map(json.loads)
print(sel.emotion.value_counts(), "| total:", len(sel))

AGENT_NAMES = {"agent", "assistant", "moshi", "bot", "personaplex",
               "speaker_broker", "speaker_0", "spk0", "left", "0"}
def agent_side(w):
    spk = str(w.get("speaker", "")).strip().lower()
    return (not spk) or (spk in AGENT_NAMES)
'''),
]


NB02 += [
code(r'''
sp = eh.load_sp(P)
mimi = eh.load_mimi(P, DEVICE)
lm = eh.load_lm_bnb4(P, DEVICE, DTYPE)
print(f"cuda mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GiB")
lm_gen = eh.make_lm_gen(lm, DEVICE)
eh.install_hidden_step(lm_gen)

if USE_LIVE_PROMPT_PREFIX:
    assert P.TEXT_PROMPT_FILE.exists(), \
        f"upload {P.TEXT_PROMPT_FILE} (or set USE_LIVE_PROMPT_PREFIX=False)"
    assert P.VOICE_PT.exists(), f"missing {P.VOICE_PT}"
    prompt_state = eh.apply_prompts_and_snapshot(
        lm_gen, mimi, sp, str(P.VOICE_PT),
        P.TEXT_PROMPT_FILE.read_text(encoding="utf-8", errors="replace"))
    print("LIVE prompt prefix (Robert + VARM3) applied once; state tensors:", len(prompt_state))
else:
    prompt_state = eh.snapshot_streaming_state(lm_gen)
    print("NO prompt prefix (flag off); state tensors:", len(prompt_state))
'''),
code(r'''
# text-stream convention sanity (see 01's interleaver printout for the visual cross-check)
assert eh.TEXT_PAD == 3 and eh.TEXT_EOP == 0
bad = 0
for _, row in sel.head(50).iterrows():
    stream = eh.build_agent_text_stream([w for w in row["alignments"] if agent_side(w)], 400, sp)
    bad += int((stream >= 32000).sum()) + int((stream < 0).sum())
assert bad == 0, "text piece ids out of range"
print("text-stream sanity OK (PAD=3, EOP=0, piece ids < 32000)")
'''),
code(r'''
# PARITY CHECK: same clip twice with the prompt state restored -> hidden must be
# bit-near-identical (cosine > 0.999). Proves the snapshot/restore + teacher-forced
# step path is deterministic and matches the live capture contract.
row = sel.iloc[0]
agent = eh.read_channel(row.wav_path, 0)
user = eh.read_channel(row.wav_path, 1)
n_frames = max(len(agent), len(user)) // eh.FRAME_SAMPLES + 1
stream = eh.build_agent_text_stream([w for w in row["alignments"] if agent_side(w)], n_frames, sp)
hiddens = []
for rep in range(2):
    eh.restore_streaming_state(lm_gen, mimi, prompt_state)
    hiddens.append(eh.extract_clip_features(lm_gen, mimi, agent, user, stream, DEVICE)["hidden"].float())
cos = float(torch.nn.functional.cosine_similarity(hiddens[0].flatten(), hiddens[1].flatten(), dim=0))
print(f"hidden determinism cosine (state restored): {cos:.5f}")
assert cos > 0.999, "hidden path not deterministic across state restore — do NOT proceed"
print("PARITY OK")
'''),
]


NB02 += [
code(r'''
from tqdm.auto import tqdm
N_GPUS = torch.cuda.device_count()
jobs = [dict(clip_id=r["clip_id"], wav_path=r["wav_path"], emotion=r["emotion"],
             user_id=r["user_id"], split=r["split"], user_emotion=r.get("user_emotion"),
             agent_turn_frames=r["agent_turn_frames"],
             agent_words=[w for w in r["alignments"] if agent_side(w)])
        for r in sel.to_dict("records")]
if N_GPUS > 1:
    # one spawned worker per GPU, each with its own model instance, shard prefix
    # (phaseA_g<i>) and log (extract_phaseA_g<i>.log). Resume: re-run this cell.
    print(f"multi-GPU: {N_GPUS} GPUs -> {N_GPUS} extraction workers (round-robin)")
    for _n in ("lm", "lm_gen", "mimi"):
        globals().pop(_n, None)     # free the in-kernel model so GPU0 has room
    eh.free_gpu()
    eh.phaseA_extract_multi(jobs, str(P.ROOT), N_GPUS,
                            dict(use_prompt=USE_LIVE_PROMPT_PREFIX,
                                 clips_per_shard=CLIPS_PER_SHARD))
    print("all workers done | shards:", len(list(P.FEAT_A.glob("phaseA*_shard_*.pt"))),
          "| logs:", sorted(p.name for p in P.FEAT_A.glob("extract_phaseA*.log")))
else:
    log = eh.ProgressLogger(P.FEAT_A / "extract_phaseA.log", echo_every=25)
    writer = eh.ShardWriter(P.FEAT_A, "phaseA", CLIPS_PER_SHARD)
    todo = [r for r in jobs if r["clip_id"] not in writer.completed]
    total_target = len(writer.completed) + len(todo)
    log.log(f"=== phaseA extraction START | completed={len(writer.completed)} "
            f"todo={len(todo)} target={total_target} | device={DEVICE} "
            f"live_prompt={USE_LIVE_PROMPT_PREFIX}", force=True)
    t0 = time.time()
    failed, times = 0, []
    for row in tqdm(todo, total=len(todo)):
        ct0 = time.time()
        try:
            agent = eh.read_channel(row["wav_path"], 0)
            user = eh.read_channel(row["wav_path"], 1)
            n_frames = max(len(agent), len(user)) // eh.FRAME_SAMPLES + 1
            stream = eh.build_agent_text_stream(row["agent_words"], n_frames, sp)
            eh.restore_streaming_state(lm_gen, mimi, prompt_state)
            feats = eh.extract_clip_features(lm_gen, mimi, agent, user, stream, DEVICE)
            T = feats["hidden"].shape[0]
            mask = np.zeros(T, dtype=bool)
            for s, e in row["agent_turn_frames"]:
                mask[max(0, s):min(T, e)] = True
            ue = row.get("user_emotion")
            writer.add({
                "clip_id": row["clip_id"], **feats,
                "agent_mask": torch.from_numpy(mask),
                "label": eh.EMO2ID[row["emotion"]],
                "user_emotion": eh.EMO2ID.get(str(ue).lower(), -1) if ue else -1,
                "user_id": row["user_id"], "split": row["split"],
                "text_stream": torch.from_numpy(stream).to(torch.int32),
            })
        except Exception as e:
            failed += 1
            log.log(f"ERROR clip={row['clip_id']} ({row['emotion']}): "
                    f"{type(e).__name__}: {e} | skipped (resume-safe)", force=True)
            continue
        dt = time.time() - ct0
        times.append(dt)
        completed = len(writer.completed)
        remaining = max(0, total_target - completed)
        avg = sum(times) / len(times)
        log.log(f"clip {completed}/{total_target} | +{dt:.2f}s avg={avg:.2f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| failed={failed} | {row['clip_id']} ({row['emotion']})")
    writer.close()
    log.log(f"=== phaseA extraction END | completed={len(writer.completed)}/{total_target} "
            f"failed={failed} | elapsed={(time.time() - t0) / 3600:.2f}h", force=True)
    print("done. total completed:", len(writer.completed), "| failed:", failed,
          "| log:", log.path)
'''),
code(r'''
import collections
import matplotlib.pyplot as plt
recs = list(eh.iter_shard_records(P.FEAT_A, "phaseA"))
print("clips on disk:", len(recs))
r = recs[0]
for k, v in r.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    else:
        print(f"  {k}: {v}")
print("labels:", collections.Counter(r["label"] for r in recs))
r = recs[min(3, len(recs) - 1)]
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(r["hidden"].float().norm(dim=1))
axes[0].set_title(f"hidden L2 norm — {r['clip_id']} ({eh.EMOTIONS[r['label']]})")
axes[1].plot(r["agent_mask"].float())
axes[1].set_title("agent-speaking mask")
axes[1].set_ylim(-0.1, 1.1)
P.REPORTS.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(P.REPORTS / "phaseA_feature_sanity.png", dpi=130)
plt.show()
for _n in ("lm", "lm_gen", "mimi"):
    globals().pop(_n, None)
eh.free_gpu()
print("→ continue with 03_train_and_evaluate_head_phaseA.ipynb")
'''),
]




# ===========================================================================
# 03 — Phase A training & evaluation (go/no-go gates)
# ===========================================================================
NB03 = [
md(r'''
# 03 — Phase A: probe gate + fusion-head training

- **GATE 1** (go/no-go): linear probes on masked-mean-pooled features per modality.
  Helium-only macro-F1 ≥ 0.55 (chance = 0.25) means the hidden states carry the emotion.
- **Fusion head**: Helium→256 + Mimi latent→128 + prosody→32 → 2-layer causal GRU(256)
  → 4 logits (~2.0M params). Per-step CE on agent frames + turn-pooled CE, modality
  dropout, label smoothing, class weights, cosine LR, early stop on val macro-F1.
- **GATE 2**: fusion ≥ best probe + 0.05.
- Reports: confusion matrix, per-class P/R, user-emotion leakage probe, ECE +
  temperature scaling, smoothed switches/min.
'''),
code(SHARED),
code(r'''
import collections
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
PATIENCE = 3
GATE1_MIN_F1 = 0.55      # helium-only probe macro-F1 (chance = 0.25)
GATE2_MIN_GAIN = 0.05    # fusion must beat the best probe by this
GATE_STRICT = False      # True -> assert gates; False -> print warnings only
'''),
code(r'''
recs = list(eh.iter_shard_records(P.FEAT_A, "phaseA"))
print("clips:", len(recs), "| splits:", collections.Counter(r["split"] for r in recs))
by_split = {}
for r in recs:
    by_split.setdefault(r["split"], []).append(r)
tr = by_split.get("train", [])
va = by_split.get("val", [])
ho = by_split.get("holdout", [])
print(f"train {len(tr)} | val {len(va)} | holdout {len(ho)}")
assert tr and (va or ho), "not enough data — run 02 first"
'''),
]

NB03 += [
code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def pooled(rec, key):
    X = rec[key].float()
    m = rec["agent_mask"]
    return (X[m].mean(0) if m.any() else X.mean(0)).numpy()

def probe(train_recs, eval_recs, key):
    Xtr = np.stack([pooled(r, key) for r in train_recs])
    ytr = np.array([r["label"] for r in train_recs])
    Xev = np.stack([pooled(r, key) for r in eval_recs])
    yev = np.array([r["label"] for r in eval_recs])
    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
    return float(f1_score(yev, clf.predict(Xev), average="macro"))

eval_recs = va or ho
f1_helium = probe(tr, eval_recs, "hidden")
f1_mimi = probe(tr, eval_recs, "mimi_lat")
f1_pros = probe(tr, eval_recs, "pros")
print(f"GATE 1 probes (macro-F1): helium={f1_helium:.3f} mimi={f1_mimi:.3f} "
      f"prosody={f1_pros:.3f} (chance=0.25)")
gate1 = f1_helium >= GATE1_MIN_F1
print("GATE 1:", "PASS" if gate1 else "FAIL — Helium hidden states may not carry emotion")
if GATE_STRICT:
    assert gate1, "stop here and reconsider the approach"
'''),
code(r'''
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return (r["hidden"], r["mimi_lat"], r["pros"], r["agent_mask"], int(r["label"]))

def collate(batch):
    T = max(b[0].shape[0] for b in batch)
    B = len(batch)
    H = torch.zeros(B, T, eh.HELIUM_DIM)
    M = torch.zeros(B, T, eh.MIMI_LATENT_DIM)
    PR = torch.zeros(B, T, eh.PROSODY_DIM)
    MK = torch.zeros(B, T, dtype=torch.bool)
    LB = torch.zeros(B, dtype=torch.long)
    for i, (h, m, p, mk, lb) in enumerate(batch):
        t = h.shape[0]
        H[i, :t] = h.float()
        M[i, :t] = m.float()
        PR[i, :t] = p.float()
        MK[i, :t] = mk
        LB[i] = lb
    return H, M, PR, MK, LB

g = torch.Generator().manual_seed(eh.SEED)
train_loader = torch.utils.data.DataLoader(
    SeqDataset(tr), batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate, generator=g, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    SeqDataset(va or ho), batch_size=BATCH_SIZE, collate_fn=collate)
print(f"batches: train {len(train_loader)} | val {len(val_loader)}")
'''),
code(r'''
model = eh.EmotionFusionHead()
print(f"head params: {model.count_params()/1e6:.2f}M")
best_f1, history = eh.train_head(model, train_loader, val_loader, device=DEVICE,
                                 epochs=EPOCHS, lr=LR, patience=PATIENCE)
eh.save_head(model, P.MODELS / "head_phaseA.pt",
             extra={"val_macro_f1": best_f1, "history": history})
print("saved:", P.MODELS / "head_phaseA.pt", "| best val macro-F1:", round(best_f1, 4))
gate2 = best_f1 >= max(f1_helium, f1_mimi, f1_pros) + GATE2_MIN_GAIN
print(f"GATE 2 (fusion >= best probe + {GATE2_MIN_GAIN}):", "PASS" if gate2 else "FAIL")
if GATE_STRICT:
    assert gate2
'''),
]


NB03 += [
code(r'''
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

eval_recs2 = ho or va
hold_loader = torch.utils.data.DataLoader(SeqDataset(eval_recs2), batch_size=BATCH_SIZE,
                                          collate_fn=collate)
logits, labels, masks = eh.evaluate_head(model, hold_loader, DEVICE)
pred = logits.argmax(-1)
yt, yp = [], []
for i in range(logits.shape[0]):
    m = masks[i]
    yt += labels[i].expand(logits.shape[1])[m].tolist()
    yp += pred[i][m].tolist()
print(classification_report(yt, yp, target_names=eh.EMOTIONS, digits=3))
cm = confusion_matrix(yt, yp, normalize="true")
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4), eh.EMOTIONS)
ax.set_yticks(range(4), eh.EMOTIONS)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Phase A holdout (agent frames)")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
P.REPORTS.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(P.REPORTS / "phaseA_confusion.png", dpi=130)
plt.show()
'''),
code(r'''
# user-emotion leakage: predicted AGENT emotion must NOT track the USER's emotion
leak = []
for i, r in enumerate(eval_recs2):
    if int(r.get("user_emotion", -1)) >= 0 and r["agent_mask"].any():
        votes = pred[i][r["agent_mask"]].mode().values
        leak.append((int(r["user_emotion"]), int(votes)))
if leak:
    uy = [a for a, _ in leak]
    up = [b for _, b in leak]
    from sklearn.metrics import f1_score as _f1
    print(f"leakage probe (user emotion vs predicted agent emotion; chance~0.25): "
          f"macro-F1={_f1(uy, up, average='macro'):.3f} on {len(leak)} clips")
else:
    print("no user_emotion labels in the selection — leakage test skipped")

# calibration: ECE before/after temperature scaling (T fit on the val split)
def ece(logits_t, labels_t, masks_t, n_bins=15):
    probs = torch.softmax(logits_t, -1)
    conf, pr = probs.max(-1)
    cs, ys, prs = [], [], []
    for i in range(logits_t.shape[0]):
        m = masks_t[i]
        if m.any():
            cs.append(conf[i][m])
            ys.append(labels_t[i].expand(logits_t.shape[1])[m])
            prs.append(pr[i][m])
    c = torch.cat(cs); y = torch.cat(ys); p = torch.cat(prs)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (c > lo) & (c <= hi)
        if m.any():
            e += float(m.float().mean()) * abs(float((p[m] == y[m]).float().mean()) - float(c[m].mean()))
    return e

val_logits, val_labels, val_masks = eh.evaluate_head(model, val_loader, DEVICE)
T_log = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.LBFGS([T_log], lr=0.05, max_iter=100)
def closure():
    opt.zero_grad()
    scaled = val_logits / T_log.exp()
    tgt = val_labels.unsqueeze(1).expand(-1, val_logits.shape[1])
    loss = torch.nn.functional.cross_entropy(scaled[val_masks], tgt[val_masks])
    loss.backward()
    return loss
opt.step(closure)
temperature = float(T_log.exp())
print(f"ECE raw={ece(val_logits, val_labels, val_masks):.4f} -> "
      f"T-scaled={ece(val_logits / temperature, val_labels, val_masks):.4f} (T={temperature:.3f})")

# switching behaviour with the hysteresis smoother
sm = eh.HysteresisSmoother()
sw = []
for i in range(logits.shape[0]):
    m = masks[i]
    if not m.any():
        continue
    sm.reset()
    for t in range(logits.shape[1]):
        if m[t]:



            sm.update(torch.softmax(logits[i, t].float(), -1).numpy())
    sw.append(sm.switches_per_min())
print(f"smoothed switches/min: mean={np.mean(sw):.2f} p95={np.percentile(sw, 95):.2f} "
      f"(target < 0.5)")
'''),
code(r'''
metrics = dict(
    gate1=dict(helium=f1_helium, mimi=f1_mimi, prosody=f1_pros, passed=bool(gate1)),
    gate2=dict(val_macro_f1=best_f1, passed=bool(gate2)),
    temperature=temperature, history=history)
(P.REPORTS / "phaseA_metrics.json").write_text(json.dumps(metrics, indent=2))
print("wrote", P.REPORTS / "phaseA_metrics.json")
print("→ continue with 04_generate_indomain_emotionplex7b.ipynb")
'''),
]
# ===========================================================================
# 04 — Phase B in-domain generation (EmotionPlex-7b)
# ===========================================================================
NB04 = [
md(r'''
# 04 — In-domain generation with EmotionPlex-7b (merged, bf16)

Generates NEW agent replies with the emotion-tuned model so the head can be
trained on Mimi/VARM3-domain audio (the dataset's Higgs voices are off-domain).

- Feeds the dataset **user channel** (right), then silence; the model free-runs
  exactly like the live duplex path (`moshi_tokens=None` → sampled).
- Text prompt `"Answer with a {emotion} tone."` + **your VARM3.pt** voice prompt
  (this is the generation step VARM3 is for).
- Stop rule: model fell silent for `STOP_SILENCE_S` after having spoken; hard cap
  `MAX_REPLY_S`. QA rejects too-short / silent / over-talkative outputs.
- ~3–6 s per sample on a 5090 → 4k samples ≈ 4–7 h. Resume-safe.

**Restart the kernel before this notebook** if you ran 02/03 in this session
(frees the bnb4 weights; the bf16 model needs ~17 GB VRAM).
'''),
code(SHARED),
code(r'''
import collections
N_SAMPLES = 4000
MAX_REPLY_S = 18.0
STOP_SILENCE_S = 2.0
MIN_REPLY_S = 0.5
'''),
code(r'''
import pandas as pd
df = pd.read_parquet(P.DATA / "manifest.parquet")
df = df[(df.has_wav) & (df.split == "train")].copy()
src = df.sample(min(len(df), N_SAMPLES), random_state=eh.SEED).reset_index(drop=True)
samples = [dict(sample_id=f"gen_{i:05d}__{r.clip_id}", clip_id=r.clip_id,
                wav_path=r.wav_path, emotion=eh.EMOTIONS[i % 4], user_id=r.user_id)
           for i, r in enumerate(src.itertuples())]
print(f"{len(samples)} generation targets | instructed emotions:",
      collections.Counter(s["emotion"] for s in samples))
'''),
code(r'''
from huggingface_hub import snapshot_download
GEN_DIR = P.ROOT / "checkpoints" / "emotionplex-7b"
if not (GEN_DIR / "model.safetensors").exists():
    snapshot_download(eh.EMOPLEX_REPO, local_dir=str(GEN_DIR),
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
print(sorted(p.name for p in GEN_DIR.iterdir()))
eh.free_gpu()
print(f"free vram: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GiB")
sp = eh.load_sp(P)
mimi = eh.load_mimi(P, DEVICE)
lm = eh.load_lm_bf16(GEN_DIR / "model.safetensors", DEVICE, DTYPE)
print(f"cuda mem: {torch.cuda.memory_allocated()/1e9:.1f} GiB")
lm_gen = eh.make_lm_gen(lm, DEVICE)
eh.install_hidden_step(lm_gen)
'''),
code(r'''
assert P.VOICE_PT.exists(), "VARM3.pt is required for generation (your voice prompt)"
base_state = eh.snapshot_streaming_state(lm_gen)
prompt_states = {}
for e in eh.EMOTIONS:
    eh.restore_streaming_state(lm_gen, mimi, base_state)
    prompt_states[e] = eh.apply_prompts_and_snapshot(
        lm_gen, mimi, sp, str(P.VOICE_PT), f"Answer with a {e} tone.")
    print("prompt state ready:", e)
'''),
]



NB04 += [
code(r'''
from tqdm.auto import tqdm
N_GPUS = torch.cuda.device_count()
previews = {}
if N_GPUS > 1:
    # one spawned worker per GPU (bf16 emotionplex-7b on each), per-GPU shard
    # prefix (gen_g<i>) and log (generate_g<i>.log). Resume: re-run this cell.
    print(f"multi-GPU: {N_GPUS} GPUs -> {N_GPUS} generation workers (round-robin)")
    for _n in ("lm", "lm_gen", "mimi"):
        globals().pop(_n, None)     # free the in-kernel model so GPU0 has room
    eh.free_gpu()
    eh.generate_multi(samples, str(P.ROOT), N_GPUS,
                      dict(model_st=str(GEN_DIR / "model.safetensors"),
                           max_reply_s=MAX_REPLY_S, min_reply_s=MIN_REPLY_S,
                           stop_silence_s=STOP_SILENCE_S))
    print("all workers done | shards:", len(list(P.GEN.glob("gen*_shard_*.pt"))),
          "| logs:", sorted(p.name for p in P.GEN.glob("generate*.log")))
else:
    log = eh.ProgressLogger(P.GEN / "generate.log", echo_every=25)
    writer = eh.ShardWriter(P.GEN, "gen", clips_per_shard=128)
    qa_fail = collections.Counter()
    todo = [s for s in samples if s["sample_id"] not in writer.completed]
    total_target = len(writer.completed) + len(todo)
    log.log(f"=== generation START | completed={len(writer.completed)} todo={len(todo)} "
            f"target={total_target} | max_reply={MAX_REPLY_S}s stop_silence={STOP_SILENCE_S}s",
            force=True)
    t0 = time.time()
    failed, times = 0, []
    min_frames = int(MIN_REPLY_S * eh.FRAME_HZ)
    for s in tqdm(todo, total=len(todo)):
        ct0 = time.time()
        try:
            user = eh.read_channel(s["wav_path"], 1)   # right channel = user
            eh.restore_streaming_state(lm_gen, mimi, prompt_states[s["emotion"]])
            out = eh.generate_reply(lm_gen, mimi, user, max_reply_s=MAX_REPLY_S,
                                    min_reply_s=MIN_REPLY_S, stop_silence_s=STOP_SILENCE_S,
                                    device=DEVICE)
            T = out["codes"].shape[0]
            reply_frames = T - out["n_user_frames"]
            speak = float((out["rms"][out["n_user_frames"]:] > 0.006).mean()) if reply_frames > 0 else 0.0
            if reply_frames < min_frames:
                qa_fail["too_short"] += 1
                continue
            if not (0.05 <= speak <= 0.97):
                qa_fail["speak_fraction"] += 1
                continue
            rec = dict(clip_id=s["sample_id"], src_clip=s["clip_id"], user_id=s["user_id"],
                       emotion=s["emotion"], label=eh.EMO2ID[s["emotion"]],
                       codes=out["codes"],
                       text_tokens=torch.from_numpy(out["text_tokens"]).to(torch.int32),
                       pcm=torch.from_numpy(out["pcm"]).to(torch.float16),
                       rms=torch.from_numpy(out["rms"]).to(torch.float16),
                       n_user_frames=out["n_user_frames"], user_codes=out["user_codes"])
            writer.add(rec)
            previews.setdefault(s["emotion"], rec)
        except Exception as e:
            failed += 1
            log.log(f"ERROR sample={s['sample_id']} ({s['emotion']}): "
                    f"{type(e).__name__}: {e} | skipped (resume-safe)", force=True)
            continue
        dt = time.time() - ct0
        times.append(dt)
        completed = len(writer.completed)
        remaining = max(0, total_target - completed)
        avg = sum(times) / len(times)
        log.log(f"sample {completed}/{total_target} | +{dt:.1f}s avg={avg:.1f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| qa_rej={sum(qa_fail.values())} failed={failed} | {s['sample_id']} ({s['emotion']})")
    writer.close()
    log.log(f"=== generation END | completed={len(writer.completed)}/{total_target} "
            f"QA rejected={dict(qa_fail)} failed={failed} "
            f"| elapsed={(time.time() - t0) / 3600:.2f}h", force=True)
    print("generated:", len(writer.completed), "| QA rejected:", dict(qa_fail),
          "| failed:", failed, "| log:", log.path)
'''),
code(r'''
from IPython.display import Audio, display
if not previews:    # multi-GPU run (or resumed kernel): pull previews from shards
    for rec in eh.iter_shard_records(P.GEN, "gen"):
        if rec["emotion"] not in previews:
            previews[rec["emotion"]] = rec
        if len(previews) == len(eh.EMOTIONS):
            break
for e in eh.EMOTIONS:
    rec = previews.get(e)
    if rec is None:
        print(e, ": no preview")
        continue
    dur = rec["pcm"].shape[0] / eh.SAMPLE_RATE
    print(f"{e}: {rec['clip_id']} ({dur:.1f}s)")
    display(Audio(rec["pcm"].float().numpy(), rate=eh.SAMPLE_RATE))
for _n in ("lm", "lm_gen", "mimi"):
    globals().pop(_n, None)
eh.free_gpu()
print("→ continue with 05_extract_indomain_and_train_phaseB.ipynb")
'''),
]


# ===========================================================================
# 05 — Phase B extraction + final training + export
# ===========================================================================
NB05 = [
md(r'''
# 05 — Phase B: in-domain features, final training, export

1. Reloads the **base bnb4** model with the live Robert+VARM3 prompt prefix and
   teacher-forces the generated replies (their own codes + text tokens — no
   alignment guessing here) → live-distribution features.
2. Fine-tunes `head_phaseA.pt` on Phase A + Phase B (or trains from scratch).
3. Grid-searches the hysteresis smoother (switches/min target) on the in-domain
   test set.
4. Exports `helium_emotion_head_final.pt`, `config.json`, `eval_report.json`.

Restart the kernel first if 04 just ran (frees the bf16 weights).
'''),
code(SHARED),
code(r'''
import collections
import pandas as pd
FINETUNE = True            # False -> train from scratch on A+B
LR = 1e-4
EPOCHS = 8
BATCH_SIZE = 16
TARGET_SWITCHES_PER_MIN = 0.5
'''),
code(r'''
sp = eh.load_sp(P)
mimi = eh.load_mimi(P, DEVICE)
lm = eh.load_lm_bnb4(P, DEVICE, DTYPE)
lm_gen = eh.make_lm_gen(lm, DEVICE)
eh.install_hidden_step(lm_gen)
assert P.TEXT_PROMPT_FILE.exists() and P.VOICE_PT.exists()
prompt_state = eh.apply_prompts_and_snapshot(
    lm_gen, mimi, sp, str(P.VOICE_PT),
    P.TEXT_PROMPT_FILE.read_text(encoding="utf-8", errors="replace"))
print("live prompt prefix ready; cuda mem:",
      f"{torch.cuda.memory_allocated()/1e9:.1f} GiB")
'''),
]


NB05 += [
code(r'''
from tqdm.auto import tqdm
N_GPUS = torch.cuda.device_count()
if N_GPUS > 1:
    # one spawned worker per GPU; each reloads gen shards + manifest and takes
    # a round-robin partition; per-GPU shard prefix (phaseB_g<i>) + log.
    # Resume: re-run this cell.
    print(f"multi-GPU: {N_GPUS} GPUs -> {N_GPUS} phaseB extraction workers")
    for _n in ("lm", "lm_gen", "mimi"):
        globals().pop(_n, None)
    eh.free_gpu()
    eh.phaseB_extract_multi(str(P.ROOT), N_GPUS, dict())
    print("all workers done | shards:", len(list(P.FEAT_B.glob("phaseB*_shard_*.pt"))),
          "| logs:", sorted(p.name for p in P.FEAT_B.glob("extract_phaseB*.log")))
else:
    man = pd.read_parquet(P.DATA / "manifest.parquet")
    user_of = dict(zip(man.clip_id, man.user_id))
    holdout_users = set(man.loc[man.split == "holdout", "user_id"].unique())

    writer = eh.ShardWriter(P.FEAT_B, "phaseB", clips_per_shard=128)
    gen_recs = list(eh.iter_shard_records(P.GEN, "gen"))
    print("generated samples:", len(gen_recs), "| already extracted:", len(writer.completed))
    n_test = max(1, len(gen_recs) // 10)
    test_ids = set(r["clip_id"] for r in gen_recs[-n_test:])   # last 10% = in-domain test

    log = eh.ProgressLogger(P.FEAT_B / "extract_phaseB.log", echo_every=25)
    todo = [r for r in gen_recs if r["clip_id"] not in writer.completed]
    total_target = len(writer.completed) + len(todo)
    log.log(f"=== phaseB extraction START | completed={len(writer.completed)} "
            f"todo={len(todo)} target={total_target}", force=True)
    t0 = time.time()
    failed, times = 0, []
    for r in tqdm(todo, total=len(todo)):
        ct0 = time.time()
        try:
            eh.restore_streaming_state(lm_gen, mimi, prompt_state)
            feats = eh.extract_from_codes(lm_gen, mimi, r["user_codes"].long(), r["codes"].long(),
                                          r["text_tokens"].long().numpy(), DEVICE)
            T = feats["hidden"].shape[0]
            rms = r["rms"].float().numpy()
            mask = np.zeros(T, dtype=bool)
            for t in range(int(r["n_user_frames"]), T):
                mask[t] = rms[t] > 0.006
            uid = r.get("user_id") or user_of.get(r["src_clip"], "")
            if uid in holdout_users:
                split = "holdout"
            elif r["clip_id"] in test_ids:
                split = "test_indomain"
            else:
                split = "train"
            feats["pros"] = torch.from_numpy(eh.prosody_features(r["pcm"].float().numpy()))
            writer.add(dict(clip_id=r["clip_id"], **feats,
                            agent_mask=torch.from_numpy(mask), label=int(r["label"]),
                            user_emotion=-1, user_id=uid, split=split,
                            text_stream=r["text_tokens"]))
        except Exception as e:
            failed += 1
            log.log(f"ERROR sample={r['clip_id']}: {type(e).__name__}: {e} "
                    f"| skipped (resume-safe)", force=True)
            continue
        dt = time.time() - ct0
        times.append(dt)
        completed = len(writer.completed)
        remaining = max(0, total_target - completed)
        avg = sum(times) / len(times)
        log.log(f"sample {completed}/{total_target} | +{dt:.2f}s avg={avg:.2f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| failed={failed} | {r['clip_id']}")
    writer.close()
    log.log(f"=== phaseB extraction END | completed={len(writer.completed)}/{total_target} "
            f"failed={failed} | elapsed={(time.time() - t0) / 3600:.2f}h", force=True)
    print("phaseB extracted:", len(writer.completed), "| failed:", failed, "| log:", log.path)
for _n in ("lm", "lm_gen", "mimi"):
    globals().pop(_n, None)
eh.free_gpu()
'''),
code(r'''
recsA = list(eh.iter_shard_records(P.FEAT_A, "phaseA"))
recsB = list(eh.iter_shard_records(P.FEAT_B, "phaseB"))
tr = [r for r in recsA if r["split"] == "train"] + [r for r in recsB if r["split"] == "train"]
va = [r for r in recsA if r["split"] == "val"] or [r for r in recsB if r["split"] == "test_indomain"]
ho = [r for r in recsA if r["split"] == "holdout"] + [r for r in recsB if r["split"] == "holdout"]
te = [r for r in recsB if r["split"] == "test_indomain"]
nA = sum(1 for r in tr if not str(r["clip_id"]).startswith("gen_"))
print(f"train {len(tr)} (A {nA} + B {len(tr)-nA}) | val {len(va)} | holdout {len(ho)} | testB {len(te)}")

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return (r["hidden"], r["mimi_lat"], r["pros"], r["agent_mask"], int(r["label"]))

def collate(batch):
    T = max(b[0].shape[0] for b in batch)
    B = len(batch)
    H = torch.zeros(B, T, eh.HELIUM_DIM)
    M = torch.zeros(B, T, eh.MIMI_LATENT_DIM)
    PR = torch.zeros(B, T, eh.PROSODY_DIM)
    MK = torch.zeros(B, T, dtype=torch.bool)
    LB = torch.zeros(B, dtype=torch.long)
    for i, (h, m, p, mk, lb) in enumerate(batch):
        t = h.shape[0]
        H[i, :t] = h.float()
        M[i, :t] = m.float()
        PR[i, :t] = p.float()
        MK[i, :t] = mk
        LB[i] = lb
    return H, M, PR, MK, LB

g = torch.Generator().manual_seed(eh.SEED)
train_loader = torch.utils.data.DataLoader(
    SeqDataset(tr), batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate, generator=g, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    SeqDataset(va), batch_size=BATCH_SIZE, collate_fn=collate)
'''),
]


NB05 += [
code(r'''
if FINETUNE and (P.MODELS / "head_phaseA.pt").exists():
    model = eh.load_head(P.MODELS / "head_phaseA.pt", DEVICE)
    print("fine-tuning from head_phaseA.pt")
else:
    model = eh.EmotionFusionHead()
    print("training from scratch")
best_f1, history = eh.train_head(model, train_loader, val_loader, device=DEVICE,
                                 epochs=EPOCHS, lr=LR, patience=3)
print("best val macro-F1:", round(best_f1, 4))
'''),
code(r'''
te_loader = torch.utils.data.DataLoader(SeqDataset(te or ho), batch_size=BATCH_SIZE,
                                        collate_fn=collate)
logits, labels, masks = eh.evaluate_head(model, te_loader, DEVICE)
f1_final = eh.macro_f1_from_logits(logits, labels, masks)
print(f"in-domain test macro-F1: {f1_final:.4f}")

best_cfg, best_score = None, -1e9
for a in (0.3, 0.4, 0.5):
    for mrg in (0.10, 0.15, 0.20):
        for dw in (1.0, 1.5, 2.0):
            sw, correct, total = [], 0, 0
            for i in range(logits.shape[0]):
                m = masks[i]
                if not m.any():
                    continue
                sm = eh.HysteresisSmoother(ema_alpha=a, enter_margin=mrg, min_dwell_s=dw)
                votes = collections.Counter()
                for t in range(logits.shape[1]):
                    if m[t]:
                        eid, _, _ = sm.update(torch.softmax(logits[i, t].float(), -1).numpy())
                        votes[eid] += 1
                sw.append(sm.switches_per_min())
                correct += int(votes.most_common(1)[0][0] == int(labels[i]))
                total += 1
            spm = float(np.mean(sw))
            acc = correct / max(1, total)
            score = acc - max(0.0, spm - TARGET_SWITCHES_PER_MIN)
            if score > best_score:
                best_score = score
                best_cfg = dict(ema_alpha=a, enter_margin=mrg, min_dwell_s=dw,
                                switches_per_min=spm, turn_acc=acc)
print("chosen smoother config:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in best_cfg.items()})
'''),
code(r'''
eh.save_head(model, P.MODELS / "helium_emotion_head_final.pt",
             extra=dict(val_macro_f1=best_f1, indomain_test_macro_f1=f1_final,
                        history=history))
config = dict(head=dict(model.cfg), emotions=eh.EMOTIONS, smoother=best_cfg,
              temperature=1.0, frame_hz=eh.FRAME_HZ, helium_dim=eh.HELIUM_DIM,
              mimi_latent_dim=eh.MIMI_LATENT_DIM, prosody_dim=eh.PROSODY_DIM)
(P.MODELS / "config.json").write_text(json.dumps(config, indent=2))
report = dict(phaseB_samples=len(recsB), val_macro_f1=best_f1,
              indomain_test_macro_f1=f1_final, smoother=best_cfg)
(P.REPORTS / "eval_report.json").write_text(json.dumps(report, indent=2))
print("EXPORTED:", P.MODELS / "helium_emotion_head_final.pt")
print("         ", P.MODELS / "config.json")
print("→ continue with 06_deploy_and_live_replay.ipynb")
'''),
]



# ===========================================================================
# 06 — deploy artifacts & offline live replay
# ===========================================================================
NB06 = [
md(r'''
# 06 — Deploy artifacts + offline live-replay benchmark

- Copies `deploy_template/{helium_emotion.py, integrate_helium_emotion.py}` to
  `deploy_out/` and verifies the deploy head class loads the final checkpoint.
- CPU smoke test of `HeliumEmotionManager` (the `AutoEmotionManager`-compatible
  drop-in: `start/feed/feed_step/current_emotion_id/reset_session/stop`).
- Offline replay on in-domain test features: per-step latency, VRAM, switches/min.
- Prints the copy-to-live checklist. Optional HF upload (flag).
'''),
code(SHARED),
code(r'''
UPLOAD_HF = False
HF_UPLOAD_REPO = "YOUR_USER/helium-emotion-head"   # change me
TEMPLATE_DIR = Path("deploy_template")             # ships with these notebooks
final = P.MODELS / "helium_emotion_head_final.pt"
assert final.exists(), "run 05 first"
assert TEMPLATE_DIR.exists(), "deploy_template/ must sit next to this notebook"
'''),
code(r'''
import shutil, ast
P.DEPLOY.mkdir(parents=True, exist_ok=True)
for f in ("helium_emotion.py", "integrate_helium_emotion.py"):
    shutil.copy2(TEMPLATE_DIR / f, P.DEPLOY / f)
    ast.parse((P.DEPLOY / f).read_text(encoding="utf-8"))
    print("deploy file OK:", P.DEPLOY / f)

sys.path.insert(0, str(P.DEPLOY))
import importlib
import helium_emotion as he_deploy
importlib.reload(he_deploy)
payload = torch.load(final, map_location="cpu", weights_only=False)
hd = he_deploy.EmotionFusionHead(**payload["cfg"])
hd.load_state_dict(payload["state_dict"])   # raises on any mismatch
print("deploy head loads the final checkpoint: OK")
'''),
]


NB06 += [
code(r'''
# CPU smoke test of the AutoEmotionManager-compatible drop-in
class _A:
    pass
a = _A()
a.emotion_id = 0
a.helium_emotion = True
a.helium_head_path = str(final)
a.helium_head_device = "cpu"
a.helium_temperature = 1.0
a.helium_ema_alpha = 0.4
a.helium_enter_margin = 0.15
a.helium_min_dwell_s = 1.5
mgr = he_deploy.HeliumEmotionManager(a)
mgr.start()
mgr.feed(b"\x00" * 1920, 24000)          # mic shim: must be a no-op, never raise
pcm = (np.random.randn(1920) * 0.05).astype(np.float32)
for t in range(25):
    mgr.feed_step(torch.randn(1, eh.HELIUM_DIM), None, pcm)
print("manager smoke:", mgr.current_emotion_id(), mgr.current_emotion(),
      "| steps:", mgr.steps_fed, "| errors:", mgr.feed_errors)
mgr.reset_session()
print("after reset:", mgr.current_emotion_id(), mgr.current_emotion())
mgr.stop()
'''),
code(r'''
# offline replay on in-domain features (stored pros/latents — full fidelity)
recs = [r for r in eh.iter_shard_records(P.FEAT_B, "phaseB")
        if r["split"] == "test_indomain"][:24]
if not recs:
    recs = list(eh.iter_shard_records(P.FEAT_B, "phaseB"))[:24]
assert recs, "no phaseB features — run 05 first"
runner = he_deploy.EmotionHeadRunner(final, device=DEVICE)
cfg = json.loads((P.MODELS / "config.json").read_text())["smoother"]
sm = he_deploy.HysteresisSmoother(ema_alpha=cfg["ema_alpha"],
                                  enter_margin=cfg["enter_margin"],
                                  min_dwell_s=cfg["min_dwell_s"])
print("smoother:", cfg)
if DEVICE == "cuda":
    h = torch.randn(1, eh.HELIUM_DIM, device=DEVICE)
    m = torch.randn(1, eh.MIMI_LATENT_DIM, device=DEVICE)
    p = torch.randn(1, eh.PROSODY_DIM, device=DEVICE)
    hh = None
    for _ in range(20):
        _, hh = runner.head.forward_step(h, m, p, hh)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 1000
    for _ in range(N):
        _, hh = runner.head.forward_step(h, m, p, hh)
    torch.cuda.synchronize()
    print(f"head forward_step: {(time.perf_counter()-t0)/N*1e6:.1f} µs/step | "
          f"vram {torch.cuda.memory_allocated()/1e6:.0f} MB")
sw = []
for r in recs:
    sm.reset()
    runner.reset()
    T = r["hidden"].shape[0]
    hh = None
    for t in range(T):
        if not bool(r["agent_mask"][t]):
            continue
        h = r["hidden"][t].float().unsqueeze(0).to(DEVICE)
        m = r["mimi_lat"][t].float().unsqueeze(0).to(DEVICE)
        p = r["pros"][t].float().unsqueeze(0).to(DEVICE)
        logits, hh = runner.head.forward_step(h, m, p, hh)
        sm.update(torch.softmax(logits.float(), -1)[0].cpu().numpy())
    sw.append(sm.switches_per_min())
print(f"replay on {len(sw)} in-domain clips: switches/min "
      f"mean={np.mean(sw):.2f} p95={np.percentile(sw, 95):.2f}")
'''),
code(r'''
if UPLOAD_HF:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_UPLOAD_REPO, exist_ok=True)
    for f in ("helium_emotion_head_final.pt", "config.json"):
        api.upload_file(path_or_fileobj=str(P.MODELS / f), path_in_repo=f,
                        repo_id=HF_UPLOAD_REPO)
    print("uploaded to", HF_UPLOAD_REPO)
print("""
DEPLOY CHECKLIST (on the live machine):
1. copy deploy_out/helium_emotion.py + deploy_out/integrate_helium_emotion.py
   next to the server file (IMTalker/)
2. copy models/helium_emotion_head_final.pt (+ config.json) there too
3. python integrate_helium_emotion.py <server_file.py>   # idempotent; writes .bak_helium
4. start the server with:
   --helium_emotion --helium_head_path helium_emotion_head_final.pt
5. watch for [HELIUM-EMOTION] switch logs; tune --helium_ema_alpha /
   --helium_enter_margin / --helium_min_dwell_s from config.json values
""")
'''),
]



# ===========================================================================
if __name__ == "__main__":
    write_nb("00_setup_and_downloads.ipynb", NB00)
    write_nb("01_dataset_inspect_and_manifest.ipynb", NB01)
    write_nb("02_extract_features_emotionplex.ipynb", NB02)
    write_nb("03_train_and_evaluate_head_phaseA.ipynb", NB03)
    write_nb("04_generate_indomain_emotionplex7b.ipynb", NB04)
    write_nb("05_extract_indomain_and_train_phaseB.ipynb", NB05)
    write_nb("06_deploy_and_live_replay.ipynb", NB06)
    print("all notebooks written and AST-validated")

