"""Shared helpers for the Helium emotion-head offline pipeline (RunPod notebooks).

Every function mirrors the LIVE server contract, verified against:
  - IMTalker/liveTry.py: model load (get_mimi / get_moshi_lm quantize_4bit=True,
    signature-filtered kwargs), prompt application (load_voice_prompt_embeddings
    -> wrapped "<system>" text tokens -> step_system_prompts(mimi)), LMGen
    construction with cfg_coef/condition_tensors + TypeError fallback.
  - IMTalker/deploy/8999_ab/live_personaplex_cached.py: streaming-state
    snapshot/restore via _flatten_streaming_state / get_streaming_state /
    set_streaming_state_inplace.
  - liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py:
    step override (prepare_step_input -> graphed_main ->
    process_transformer_output) returning (tokens, transformer_out,
    transformer_out); tokens[1,9,1] = text at [:,0], 8 audio codes at [:,1:].
  - moshi modules: mimi.encode -> [B,8,T] codes; quantizer.decode -> [B,512,T]
    (stateless RRVQ decode).
"""
from __future__ import annotations

import contextlib
import gc
import inspect
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (verified against moshi loaders.py / lm.py / the live server)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
FRAME_HZ = 12.5
FRAME_SAMPLES = 1920            # mimi frame_size; the live server asserts this
N_AUDIO_CODEBOOKS = 8
TEXT_PAD = 3                    # zero_text_code / padding token
TEXT_EOP = 0                    # end-of-padding / "ungenerated" token
HELIUM_DIM = 4096
MIMI_LATENT_DIM = 512
PROSODY_DIM = 6
SEED = 1234

# id order MUST match the live server (auto_emotion.py):
# EMOTION_NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", ...}
EMOTIONS = ["neutral", "happy", "sad", "angry"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}

BNB4_REPO = "brianmatzelle/personaplex-7b-v1-bnb-4bit"
NVIDIA_REPO = "nvidia/personaplex-7b-v1"            # mimi/spm/voices.tgz (gated)
DATASET_REPO = "frabonzi/emotionplex-duplex-multispeaker"  # public
EMOPLEX_REPO = "frabonzi/emotionplex-7b"


# ---------------------------------------------------------------------------
# Run mode (FULL vs SMOKE_TEST). The flag is set ONCE in NB00's first cell
# (which writes <root>/run_config.json); every other notebook — and every
# spawned GPU worker — reads it back through get_paths(), so the whole
# pipeline resolves the same mode without any per-notebook edits.
# ---------------------------------------------------------------------------
SMOKE_DIRNAME = "_smoke"
RUN_CONFIG_NAME = "run_config.json"


def write_run_mode(root, smoke: bool) -> dict:
    """Persist the run mode for this root (NB00's first cell calls this)."""
    cfg = {"smoke_test": bool(smoke), "version": 1,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = Path(root) / RUN_CONFIG_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    return cfg


def read_smoke_mode(root=None) -> bool:
    """Mode resolution order: EMOTION_HEAD_SMOKE env var (1/0; automation
    override only) > <root>/run_config.json (written by NB00) > FULL (default;
    a missing file always means the full pipeline)."""
    env = os.environ.get("EMOTION_HEAD_SMOKE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    root = Path(root or os.environ.get("EMOTION_HEAD_ROOT", "/workspace/emotion_head"))
    cfg = root / RUN_CONFIG_NAME
    if cfg.exists():
        try:
            return bool(json.loads(cfg.read_text()).get("smoke_test", False))
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_paths(root=None, smoke=None) -> SimpleNamespace:
    """ROOT holds the SHARED assets (checkpoints, prompts, raw dataset).
    Derived artifacts (manifest, features, gen, models, reports, deploy_out)
    live under OUT — which is ROOT itself in full mode (legacy layout, fully
    compatible with existing runs) and ROOT/_smoke in smoke mode, so a smoke
    run can never touch full-run artifacts."""
    root = Path(root or os.environ.get("EMOTION_HEAD_ROOT", "/workspace/emotion_head"))
    if smoke is None:
        smoke = read_smoke_mode(root)
    ckpt = root / "checkpoints" / "personaplex_bnb4"
    out = root / SMOKE_DIRNAME if smoke else root
    return SimpleNamespace(
        ROOT=root,
        SMOKE=bool(smoke),
        OUT=out,
        CKPT=ckpt,
        MOSHI_BNB4=ckpt / "model_bnb_4bit.pt",
        MIMI_FILE=ckpt / "tokenizer-e351c8d8-checkpoint125.safetensors",
        SPM_FILE=ckpt / "tokenizer_spm_32k_3.model",
        VOICE_PT=ckpt / "voices" / "VARM3.pt",
        MOSHI_ROOT=ckpt / "moshi",
        PROMPTS=root / "prompts",
        TEXT_PROMPT_FILE=root / "prompts" / "RB_Robert_System_Prompt_full.txt",
        DATA=root / "data" / "emotionplex",   # raw downloads (shared by both modes)
        MANIFEST=out / "data" / "emotionplex" / "manifest.parquet",
        GEN=out / "data" / "indomain",
        FEAT_A=out / "features" / "phaseA",
        FEAT_B=out / "features" / "phaseB",
        MODELS=out / "models",
        REPORTS=out / "reports",
        DEPLOY=out / "deploy_out",
    )


def mode_banner(P) -> None:
    """Loud per-notebook mode banner (called by the shared setup cell)."""
    if P.SMOKE:
        msg = f"MODE: SMOKE_TEST — tiny end-to-end verification; outputs -> {P.OUT}"
    else:
        msg = f"MODE: FULL — outputs -> {P.ROOT}"
    bar = "=" * max(60, len(msg) + 2)
    print(bar)
    print(msg)
    print(bar)


def run_profile(paths=None) -> SimpleNamespace:
    """Every size knob of the pipeline in one place. The FULL values are the
    historical hard-coded ones (unchanged behaviour); SMOKE shrinks data,
    epochs and steps for a fast end-to-end verification. Notebook parameter
    cells read these (`RUN.nb03_epochs` etc.) so a value can still be
    hand-overridden in the notebook when needed."""
    smoke = paths.SMOKE if paths is not None else read_smoke_mode()
    full = dict(
        # NB00 downloads (smoke: stratified subset chosen from a sidecar pool)
        nb00_download_all_audio=True, nb00_max_clips_subset=8000,
        smoke_sidecar_candidates_train=0, smoke_sidecar_candidates_val=0,
        smoke_train_clips_per_emotion=0, smoke_val_clips_per_emotion=0,
        # NB02 phase A extraction (eval = per-split-per-emotion cap, val+holdout)
        nb02_max_clips=8000, nb02_eval_clips_per_emotion=200, nb02_clips_per_shard=256,
        # NB03 phase A training
        nb03_batch_size=16, nb03_epochs=12, nb03_patience=3,
        # NB04 in-domain generation
        nb04_n_samples=4000, nb04_max_reply_s=18.0, nb04_stop_silence_s=2.0,
        nb04_min_reply_s=0.5, nb04_gen_clips_per_shard=128, nb04_max_attempts=2,
        # NB05 phase B extraction + training
        nb05_epochs=8, nb05_batch_size=16, nb05_patience=3, nb05_clips_per_shard=128,
        # NB06 deploy/replay
        nb06_replay_clips=24, nb06_latency_iters=1000,
    )
    smoke_cfg = dict(
        nb00_download_all_audio=False, nb00_max_clips_subset=8000,
        smoke_sidecar_candidates_train=100, smoke_sidecar_candidates_val=30,
        smoke_train_clips_per_emotion=10, smoke_val_clips_per_emotion=3,
        nb02_max_clips=8000, nb02_eval_clips_per_emotion=200, nb02_clips_per_shard=8,
        nb03_batch_size=4, nb03_epochs=2, nb03_patience=2,
        nb04_n_samples=12, nb04_max_reply_s=8.0, nb04_stop_silence_s=1.0,
        nb04_min_reply_s=0.5, nb04_gen_clips_per_shard=4, nb04_max_attempts=2,
        nb05_epochs=2, nb05_batch_size=4, nb05_patience=2, nb05_clips_per_shard=8,
        nb06_replay_clips=8, nb06_latency_iters=200,
    )
    assert set(full) == set(smoke_cfg), "full/smoke profile keys diverged"
    return SimpleNamespace(**(smoke_cfg if smoke else full))


def manifest_clip_stem(row) -> str:
    """Clip id from a manifest row. The clip manifests (train/val.jsonl) carry
    a `path` like "stereo_wav/000/<id>.wav"; metadata manifests
    (dialogues.jsonl) carry a bare `id`. Returns "" when neither is present."""
    p = str(row.get("path", ""))
    return Path(p).stem if p else str(row.get("id", ""))


def ensure_smoke_subset(P, run, rows) -> set:
    """Return the smoke clip subset (set of clip stems).

    Reads the file NB00's smoke download cell wrote. If that file is missing —
    e.g. the dataset was already fully downloaded before smoke mode was
    enabled, so NB00's smoke cells were never run — derives an equivalent
    stratified subset (per-emotion, deterministic seed) from the manifest
    `rows` NB01 already parsed, and writes the file so re-runs and downstream
    notebooks stay consistent. No downloads happen here.
    """
    subset_file = P.OUT / "smoke_subset.json"
    if subset_file.exists():
        return set(json.loads(subset_file.read_text())["clips"])
    print(f"SMOKE: {subset_file} missing (NB00 smoke download cells not run) — "
          "deriving the stratified subset from the local manifests instead")
    has_path = any(str(r.get("path", "")) for r in rows)
    rng = np.random.RandomState(SEED)
    keep = set()
    for want_val, per_emo in ((False, run.smoke_train_clips_per_emotion),
                              (True, run.smoke_val_clips_per_emotion)):
        by_emo = {e: set() for e in EMOTIONS}
        for r in rows:
            if has_path and not str(r.get("path", "")):
                continue                       # metadata manifests (dialogues.jsonl)
            is_val = "val" in str(r.get("_manifest", "")).lower()
            if is_val != want_val:
                continue
            stem = manifest_clip_stem(r)
            e = str(r.get("res_emotion") or "").lower()
            if e not in EMO2ID:
                parts = stem.split("_")        # "<user>_<agent>_NNN"
                e = parts[1] if len(parts) > 1 else ""
            if stem and e in EMO2ID:
                by_emo[e].add(stem)
        for e in EMOTIONS:
            cand = sorted(by_emo[e])
            take = [cand[i] for i in rng.permutation(len(cand))[: min(per_emo, len(cand))]]
            keep.update(take)
            print(f"  smoke fallback {'val' if want_val else 'train'} {e}: "
                  f"picked {len(take)}/{len(cand)}")
    if not keep:
        raise RuntimeError(
            "smoke subset fallback found no clips — run NB00's download cells "
            "(or check the manifest row schema: expected 'path' or 'id')")
    P.OUT.mkdir(parents=True, exist_ok=True)
    subset_file.write_text(json.dumps(dict(clips=sorted(keep)), indent=2))
    print(f"SMOKE: wrote fallback subset ({len(keep)} clips) -> {subset_file}")
    return keep


def diagnose_gen_dir(P) -> str:
    """Explain why Phase-B gen shards are missing: shows what the gen dir
    actually contains, the tail of any generate*.log (the QA-reject counters
    live there), and — in smoke mode — whether the shards landed in the
    FULL-mode gen dir instead (i.e. NB04 ran with a stale full-mode kernel)."""
    lines = [f"gen dir: {P.GEN} (exists={P.GEN.exists()})"]
    shards = sorted(P.GEN.glob("gen*_shard_*.pt"))
    idx = sorted(P.GEN.glob("gen*_index.json"))
    lines.append(f"  shard files: {len(shards)} | index files: {len(idx)}")
    for lg in sorted(P.GEN.glob("generate*.log")):
        try:
            tail = lg.read_text(errors="replace").splitlines()[-4:]
        except Exception:
            continue
        lines.append(f"  tail of {lg.name}:")
        lines.extend("    " + t for t in tail)
    if P.SMOKE:
        full_gen = P.ROOT / "data" / "indomain"
        n_full = len(list(full_gen.glob("gen*_shard_*.pt"))) if full_gen.exists() else 0
        if n_full:
            lines.append(
                f"  CAUSE: the FULL-mode gen dir ({full_gen}) holds {n_full} shard(s) "
                "— NB04 was run with a full-mode kernel. Re-run NB04 with a FRESH "
                "kernel after enabling smoke mode (12 samples, a few minutes).")
        elif not shards:
            lines.append(
                "  CAUSE: no gen shards anywhere — run NB04 first (smoke mode, fresh "
                "kernel). If NB04 already ran, check its QA counters in the log "
                "above: everything may have been QA-rejected (re-run 04, and consider "
                "raising nb04_max_reply_s in eh_common.run_profile).")
    elif not shards:
        lines.append("  run NB04 first.")
    return "\n".join(lines)


def smoke_artifact_check(P) -> bool:
    """NB06 smoke finale: verify every pipeline artifact exists and print a
    clear PASS / MISSING list."""
    checks = {
        "manifest.parquet": P.MANIFEST.exists(),
        "phaseA shards": any(P.FEAT_A.glob("phaseA*_shard_*.pt")),
        "head_phaseA.pt": (P.MODELS / "head_phaseA.pt").exists(),
        "phaseA_metrics.json": (P.REPORTS / "phaseA_metrics.json").exists(),
        "gen shards": any(P.GEN.glob("gen*_shard_*.pt")),
        "phaseB shards": any(P.FEAT_B.glob("phaseB*_shard_*.pt")),
        "helium_emotion_head_final.pt": (P.MODELS / "helium_emotion_head_final.pt").exists(),
        "config.json": (P.MODELS / "config.json").exists(),
        "eval_report.json": (P.REPORTS / "eval_report.json").exists(),
        "deploy helium_emotion.py": (P.DEPLOY / "helium_emotion.py").exists(),
        "deploy integrate_helium_emotion.py": (P.DEPLOY / "integrate_helium_emotion.py").exists(),
    }
    print("smoke artifact check:")
    for k, v in checks.items():
        print(f"  {'OK      ' if v else 'MISSING '} {k}")
    ok = all(checks.values())
    print("SMOKE PIPELINE PASSED — all artifacts present" if ok
          else "SMOKE PIPELINE INCOMPLETE — see MISSING above")
    return ok


def ensure_moshi_importable(paths) -> str:
    for cand in (paths.MOSHI_ROOT, paths.CKPT, paths.ROOT):
        if Path(cand).is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    import moshi  # noqa: F401
    return str(moshi.__file__)


# ---------------------------------------------------------------------------
# Model loading (mirrors liveTry.py)
# ---------------------------------------------------------------------------
def load_sp(paths):
    import sentencepiece
    return sentencepiece.SentencePieceProcessor(model_file=str(paths.SPM_FILE))


def _filtered_kwargs(fn, candidates: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(candidates)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


def _freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class ProgressLogger:
    """Append-only run logger for long loops: every line is written (flushed)
    to a timestamped log file so progress survives kernel crashes / SSH
    disconnects, and echoed to the notebook every `echo_every` lines (START /
    END / ERROR lines should pass force=True to always print).

    File lines carry: completed / total, per-item + rolling-average seconds,
    remaining count, ETA hours, and failure counters.
    """

    def __init__(self, path, echo: bool = True, echo_every: int = 25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.echo_every = max(1, int(echo_every))
        self._n = 0

    def log(self, msg: str, force: bool = False) -> None:
        self._n += 1
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo and (force or self._n % self.echo_every == 0):
            print(line)


def hf_fetch(repo_id: str, filename: str, dst: Path) -> Path:
    """Download `filename` from HF `repo_id` to `dst` (skips if dst exists).

    Used for assets the bnb4 community repo does not ship (mimi, SentencePiece,
    voices) — those come from the gated nvidia/personaplex-7b-v1 repo.
    """
    dst = Path(dst)
    if dst.exists():
        print("already present:", dst)
        return dst
    from huggingface_hub import hf_hub_download
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id, filename)
    shutil.copy2(src, dst)
    print(f"fetched {repo_id}/{filename} -> {dst}")
    return dst


def load_mimi(paths, device="cuda"):
    from moshi.models import loaders
    mimi = loaders.get_mimi(str(paths.MIMI_FILE), device=device)
    return _freeze(mimi)


def load_lm_bnb4(paths, device="cuda", dtype=torch.bfloat16):
    """bitsandbytes-4bit PersonaPlex, exactly the live server's load path."""
    from moshi.models import loaders
    kwargs = _filtered_kwargs(
        loaders.get_moshi_lm,
        dict(device=device, dtype=dtype, quantize_4bit=True),
    )
    return _freeze(loaders.get_moshi_lm(str(paths.MOSHI_BNB4), **kwargs))


def _pt_valid(pt_path) -> bool:
    """A complete torch.save file is a zip with a central directory; a
    truncated/concurrently-written one is not (cheap check — reads the file
    tail only)."""
    import zipfile
    try:
        p = Path(pt_path)
        return p.stat().st_size > 0 and zipfile.is_zipfile(p)
    except Exception:
        return False


def ensure_pt_checkpoint(weight_path: str) -> str:
    """One-off safetensors -> .pt conversion, done ONCE (ideally in the parent
    process before spawning workers). The repo's .pt load path uses
    torch.load(map_location='cpu') — its safetensors path instead loads the
    whole ~17GB state dict onto GPU 0 (dev.type == 'cuda'), which OOMs GPU 0
    when several per-GPU workers load concurrently. Atomic (tmp file +
    os.replace) so concurrent converters can never leave a corrupt file, and
    an existing corrupt (truncated) .pt is reconverted."""
    weight_path = str(weight_path)
    pt_path = weight_path + ".pt"
    if Path(pt_path).exists() and _pt_valid(pt_path):
        return pt_path
    from safetensors.torch import load_file
    tmp_path = f"{pt_path}.tmp{os.getpid()}"
    print(f"[eh_common] converting safetensors -> {pt_path} (one-off, ~17GB)")
    torch.save(load_file(weight_path), tmp_path)
    os.replace(tmp_path, pt_path)          # atomic on POSIX and Windows
    return pt_path


def load_lm_bf16(weight_path, device="cuda", dtype=torch.bfloat16):
    """Merged emotionplex-7b (bf16). Prefers the pre-converted .pt (CPU
    map_location — no GPU-0 spike); direct safetensors load otherwise, with an
    atomic .pt conversion fallback."""
    from moshi.models import loaders
    weight_path = str(weight_path)
    pt_path = weight_path + ".pt" if weight_path.endswith(".safetensors") else None
    kwargs = _filtered_kwargs(loaders.get_moshi_lm, dict(device=device, dtype=dtype))
    if pt_path and Path(pt_path).exists() and _pt_valid(pt_path):
        return _freeze(loaders.get_moshi_lm(pt_path, **kwargs))
    try:
        return _freeze(loaders.get_moshi_lm(weight_path, **kwargs))
    except Exception:
        if not pt_path:
            raise
        return _freeze(loaders.get_moshi_lm(ensure_pt_checkpoint(weight_path), **kwargs))


def make_lm_gen(lm, device="cuda", cfg_coef=1.0):
    """LMGen construction mirrors liveTry.py L216-244 (incl. TypeError fallback)."""
    from moshi.models.lm import LMGen
    try:
        from moshi.run_inference import get_condition_tensors
        cond = get_condition_tensors("personaplex", lm, batch_size=1, cfg_coef=float(cfg_coef))
    except Exception:
        cond = {}
    try:
        lm_gen = LMGen(lm, cfg_coef=float(cfg_coef), condition_tensors=cond)
    except TypeError:
        lm_gen = LMGen(lm, device=device)
    # audio_silence_frame_cnt stays at the LMGen default: extraction and
    # generation must use the exact prompt prefix liveTry.py uses
    # (LMGen(lm, device=device) with no overrides).
    lm_gen.streaming_forever(1)
    return lm_gen


def free_gpu(*_objs) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Prompts + streaming-state snapshot (mirrors liveTry / live_personaplex_cached)
# ---------------------------------------------------------------------------
def wrap_personaplex_system_prompt(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def snapshot_streaming_state(lm_gen) -> dict:
    from moshi.modules.streaming import _flatten_streaming_state
    tensors, meta = {}, {}
    _flatten_streaming_state(tensors, meta, lm_gen.get_streaming_state(), prefix="")
    state = {k: v.detach().cpu().clone() for k, v in tensors.items()}
    state.update(meta)
    return state


def restore_streaming_state(lm_gen, mimi, state) -> None:
    lm_gen.set_streaming_state_inplace(dict(state))
    if mimi is not None:
        with contextlib.suppress(Exception):
            mimi.reset_streaming()


@torch.no_grad()
def apply_prompts_and_snapshot(lm_gen, mimi, sp, voice_pt: str, text_prompt: str) -> dict:
    """Mirror of MoshiOnlyEngine._apply_system_prompts followed by a snapshot
    of the post-prompt streaming state (prompts replayed ONCE, not per clip)."""
    voice_pt = str(voice_pt)
    if voice_pt and voice_pt.endswith(".pt") and hasattr(lm_gen, "load_voice_prompt_embeddings"):
        lm_gen.load_voice_prompt_embeddings(voice_pt)
    elif voice_pt and hasattr(lm_gen, "load_voice_prompt"):
        lm_gen.load_voice_prompt(voice_pt)
    if text_prompt and hasattr(sp, "encode"):
        lm_gen.text_prompt_tokens = sp.encode(wrap_personaplex_system_prompt(text_prompt))
    else:
        lm_gen.text_prompt_tokens = None
    lm_gen.step_system_prompts(mimi)
    with contextlib.suppress(Exception):
        mimi.reset_streaming()
    return snapshot_streaming_state(lm_gen)


# ---------------------------------------------------------------------------
# Hidden-state step override (EXACT copy of the live AH server's override,
# _install_graph_hidden_capture). Returns (tokens, transformer_out,
# transformer_out); tokens[1,9,1]: text at [:,0], 8 audio codes at [:,1:].
# ---------------------------------------------------------------------------
def install_hidden_step(lm_gen) -> None:
    if not (hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output")):
        raise RuntimeError("LMGen lacks prepare_step_input/process_transformer_output")

    @torch.no_grad()
    def personaplex_step_with_hidden(
        self_gen,
        input_tokens: torch.Tensor = None,
        moshi_tokens: torch.Tensor = None,
        text_token: torch.Tensor = None,
        depformer_replace_tokens: torch.Tensor | None = None,
    ):
        prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
        if prepared is None:
            return None
        input_, provided_, target_, model_input_position, target_position = prepared
        state = self_gen._streaming_state
        transformer_out, text_logits = state.graphed_main(input_)
        output = self_gen.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
        )
        return output, transformer_out, transformer_out

    lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)


def _step_force(lm_gen, **kwargs):
    """`lm_gen._step` that tolerates the offset-0 init step.

    moshi's prepare_step_input returns None exactly once: when
    `state.offset == 0` (first call after streaming state (re)entry). That
    call's inputs are discarded (cache[:, :, 0] is overwritten with the initial
    tokens), so re-feeding the SAME frame is exact. When the state was restored
    from a post-prompt snapshot (offset > 0) the first call already returns a
    tuple and this loop exits immediately.
    """
    out = lm_gen._step(**kwargs)
    while out is None:
        out = lm_gen._step(**kwargs)
    return out


# ---------------------------------------------------------------------------
# Mimi IO (streaming, live-identical)
# ---------------------------------------------------------------------------
@torch.no_grad()
def mimi_encode_codes(mimi, pcm: np.ndarray, device="cuda") -> torch.Tensor:
    """float32 mono pcm @24kHz -> int64 codes [T, 8] (streaming encode)."""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n_frames = max(1, math.ceil(pcm.shape[0] / FRAME_SAMPLES))
    pad = n_frames * FRAME_SAMPLES - pcm.shape[0]
    if pad:
        pcm = np.pad(pcm, (0, pad))
    frames = torch.from_numpy(pcm).view(1, 1, n_frames, FRAME_SAMPLES)
    codes = []
    with mimi.streaming(1):
        for t in range(n_frames):
            c = mimi.encode(frames[:, :, t].to(device))   # [1, K, 1]
            codes.append(c[0, :, 0].long().cpu())
    codes = torch.stack(codes)                            # [T, K]
    if codes.shape[1] != N_AUDIO_CODEBOOKS:
        codes = codes[:, :N_AUDIO_CODEBOOKS]
    return codes


@torch.no_grad()
def mimi_decode_codes(mimi, codes_t8: torch.Tensor, device="cuda") -> np.ndarray:
    """codes [T, 8] -> float32 pcm [T*1920] (streaming decode)."""
    pcm = []
    with mimi.streaming(1):
        for t in range(codes_t8.shape[0]):
            frame = codes_t8[t].view(1, N_AUDIO_CODEBOOKS, 1).to(device)
            pcm.append(mimi.decode(frame)[0, 0].float().cpu().numpy())
    return np.concatenate(pcm) if pcm else np.zeros(0, np.float32)


@torch.no_grad()
def codes_to_latent(mimi, codes_t8: torch.Tensor, device="cuda") -> torch.Tensor:
    """codes [T, 8] -> [T, 512] float32 latent. RRVQ decode is stateless
    (moshi quantization/vq.py), so this never disturbs streaming state."""
    lat = mimi.quantizer.decode(codes_t8.T.unsqueeze(0).long().to(device))
    return lat[0].T.float().cpu()


# ---------------------------------------------------------------------------
# DSP prosody (per 80ms frame) — identical implementation ships in the deploy
# module (helium_emotion.py). FFT-based autocorrelation F0 (fast).
# ---------------------------------------------------------------------------
def prosody_features(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """pcm float32 @24kHz -> [T, 6] float32:
    [log_rms, d_log_rms, log2(f0/110), voicing, zcr, spectral_centroid_norm]."""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n_frames = max(1, math.ceil(pcm.shape[0] / FRAME_SAMPLES))
    pad = n_frames * FRAME_SAMPLES - pcm.shape[0]
    if pad:
        pcm = np.pad(pcm, (0, pad))
    x = pcm.reshape(n_frames, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
    log_rms = np.log(rms + 1e-8)
    d_rms = np.concatenate([[0.0], np.diff(log_rms)]).astype(np.float32)
    # FFT autocorrelation F0, search 60..500 Hz
    min_lag = int(sr / 500)
    max_lag = int(sr / 60)
    xw = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * FRAME_SAMPLES - 1).bit_length()
    spec = np.fft.rfft(xw, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :FRAME_SAMPLES]
    denom = ac[:, 0] + 1e-12
    seg = ac[:, min_lag:max_lag]
    lag = seg.argmax(axis=1) + min_lag
    voicing = (seg.max(axis=1) / denom).astype(np.float32)
    f0 = np.where(voicing > 0.3, sr / np.maximum(lag, 1), 0.0).astype(np.float32)
    logf0 = np.where(f0 > 0, np.log2(np.maximum(f0, 1e-3) / 110.0), 0.0).astype(np.float32)
    zcr = np.mean(np.abs(np.diff(np.signbit(x), axis=1)), axis=1).astype(np.float32)
    mags = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.linspace(0.0, 1.0, mags.shape[1], dtype=np.float32)
    centroid = ((mags * freqs).sum(1) / (mags.sum(1) + 1e-12)).astype(np.float32)
    return np.stack([log_rms, d_rms, logf0, voicing, zcr, centroid], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Text stream from word alignments (moshi interleaver convention):
# TEXT_PAD (3) everywhere; TEXT_EOP (0) on the frame just before a word starts;
# the word's sentencepiece pieces from its start frame onward.
# NB01 prints the EmotionPlex interleaver for visual cross-check; NB02 asserts
# piece ids < 32000 on sample clips before the expensive extraction starts.
# ---------------------------------------------------------------------------
def build_agent_text_stream(alignments, n_frames: int, sp, fps: float = FRAME_HZ) -> np.ndarray:
    stream = np.full(int(n_frames), TEXT_PAD, dtype=np.int64)

    def _get(w, *keys):
        for k in keys:
            if isinstance(w, dict) and k in w:
                return w[k]
        return None

    for w in alignments or []:
        word = _get(w, "word", "w", "text", "token")
        start = _get(w, "start", "x0", "t0")
        if word is None or start is None:
            continue
        try:
            pieces = list(sp.encode(str(word)))
        except Exception:
            continue
        if not pieces:
            continue
        s = min(max(0, int(round(float(start) * fps))), int(n_frames) - 1)
        if s > 0 and stream[s - 1] == TEXT_PAD:
            stream[s - 1] = TEXT_EOP
        for i, p in enumerate(pieces):
            idx = s + i
            if idx >= n_frames:
                break
            stream[idx] = int(p)
    return stream


# ---------------------------------------------------------------------------
# Teacher-forced feature extraction (Phase A: dataset clips; Phase B: generated)
# ---------------------------------------------------------------------------
def _stream8(codes):
    """Keep the first 8 codebooks of a [T, K] code tensor. PersonaPlex's LM
    step returns 17 streams (1 text + 8 agent + 8 user echo); legacy gen
    shards saved all 16 audio streams — the agent stream is the first 8."""
    return codes[:, :N_AUDIO_CODEBOOKS] if int(codes.shape[1]) > N_AUDIO_CODEBOOKS else codes


@torch.no_grad()
def extract_from_codes(lm_gen, mimi, user_codes, agent_codes, text_tokens=None,
                       device="cuda") -> dict:
    """Teacher-force one conversation through the live-identical step path.

    user_codes/agent_codes: int64 [T, 8] (16-wide legacy gen codes are sliced
    to the agent stream). text_tokens: int64 [T] (default: all PAD).
    Returns hidden [T,4096] fp16 (final-layer transformer_out, post out_norm,
    the SAME tensor the live server captures), agent_codes int16, mimi_lat
    [T,512] fp16.
    """
    user_codes = _stream8(user_codes)
    agent_codes = _stream8(agent_codes)
    T = max(int(user_codes.shape[0]), int(agent_codes.shape[0]))
    sil = mimi_encode_codes(mimi, np.zeros(FRAME_SAMPLES, np.float32), device)[0]  # [8]

    def _pad(codes):
        if codes.shape[0] >= T:
            return codes[:T]
        pad = sil.unsqueeze(0).repeat(T - codes.shape[0], 1)
        return torch.cat([codes, pad], 0)

    user_codes = _pad(user_codes)
    agent_codes = _pad(agent_codes)
    if text_tokens is None:
        text_tokens = np.full(T, TEXT_PAD, dtype=np.int64)
    text_tokens = np.asarray(text_tokens, dtype=np.int64)
    if text_tokens.shape[0] < T:
        text_tokens = np.pad(text_tokens, (0, T - text_tokens.shape[0]), constant_values=TEXT_PAD)
    text_tokens = text_tokens[:T]

    hidden = torch.empty(T, HELIUM_DIM, dtype=torch.float16)
    for t in range(T):
        out = _step_force(
            lm_gen,
            input_tokens=user_codes[t].view(1, N_AUDIO_CODEBOOKS, 1).to(device),
            moshi_tokens=agent_codes[t].view(1, N_AUDIO_CODEBOOKS, 1).to(device),
            text_token=int(text_tokens[t]),
        )
        h = out[2][0, -1]
        hidden[t] = h.to(torch.float16).cpu()
    latent = codes_to_latent(mimi, agent_codes, device)
    return {
        "hidden": hidden,                               # [T, 4096] fp16
        "agent_codes": agent_codes.to(torch.int16),     # [T, 8]
        "mimi_lat": latent.to(torch.float16),           # [T, 512]
    }


@torch.no_grad()
def extract_clip_features(lm_gen, mimi, agent_pcm, user_pcm, text_stream=None,
                          device="cuda") -> dict:
    """Phase A entry point: encode both channels, then teacher-force."""
    user_codes = mimi_encode_codes(mimi, user_pcm, device)
    agent_codes = mimi_encode_codes(mimi, agent_pcm, device)
    out = extract_from_codes(lm_gen, mimi, user_codes, agent_codes, text_stream, device)
    out["pros"] = torch.from_numpy(prosody_features(agent_pcm))
    return out


# ---------------------------------------------------------------------------
# Free-running generation (Phase B, emotionplex-7b)
# ---------------------------------------------------------------------------
def _prep_user_turn(user_pcm, rms_threshold=0.006):
    """Pad to whole frames, per-frame user-channel RMS, the content-based
    user-speech-end frame (RMS VAD; floor = median frame RMS ~= roomtone, with
    a 0.32s hangover), and the quietest-1s roomtone slice of the clip (the
    recording's own background). GPU-free helper of generate_reply (unit-
    tested in _smoke_harness)."""
    user_pcm = np.asarray(user_pcm, dtype=np.float32).reshape(-1)
    n_uf = max(1, math.ceil(len(user_pcm) / FRAME_SAMPLES))
    pad = n_uf * FRAME_SAMPLES - len(user_pcm)
    if pad:
        user_pcm = np.pad(user_pcm, (0, pad))
    urms = np.sqrt((user_pcm.reshape(n_uf, FRAME_SAMPLES) ** 2).mean(axis=1) + 1e-12)
    floor = float(np.median(urms))
    speech_thr = max(2.0 * rms_threshold, min(0.03, 6.0 * floor))
    speech_frames = np.nonzero(urms > speech_thr)[0]
    if len(speech_frames):
        hang = max(1, int(0.32 * FRAME_HZ))
        user_end = min(n_uf, int(speech_frames[-1]) + hang + 1)
    else:
        user_end = n_uf                        # no speech detected -> old behavior
    win = max(1, int(FRAME_HZ))                # 1 s
    if n_uf > win:
        w = np.lib.stride_tricks.sliding_window_view(urms, win).mean(axis=1)
        idx = int(np.argmin(w))
    else:
        idx = 0
    room = user_pcm[idx * FRAME_SAMPLES:(idx + win) * FRAME_SAMPLES]
    return user_pcm, urms, user_end, room


@torch.no_grad()
def generate_reply(lm_gen, mimi, user_pcm, *, max_reply_s=20.0, min_reply_s=0.5,
                   stop_silence_s=2.0, rms_threshold=0.006, device="cuda") -> dict:
    """Feed the user channel, then looped roomtone; the model free-runs
    (moshi_tokens=None -> sampled), exactly like the live duplex path.

    Duplex reality (dataset card): each stereo clip is [user speech][gap]
    [original agent reply][padding] on ONE timeline, with roomtone mixed under
    the user channel. A duplex model therefore starts replying BEFORE the user
    clip ends, so the reply region is defined by CONTENT — everything after the
    user's speech ends (RMS VAD) — not by the clip boundary. After the clip the
    input switches to looped roomtone (the quietest 1s of the clip's own user
    channel = the recording's real background, matching the dataset's
    roomtone-mixed user channel), never digital zeros: the model only ever saw
    real mic background in training and stays silent on out-of-distribution
    pure-silence codes (the repo itself uses dedicated SINE tokens, not encoded
    zeros, for prompt-phase user silence).

    Returns agent codes [T,8] (model output on the whole timeline), per-frame
    text tokens, decoded pcm, per-frame RMS, the user-speech-end frame
    (`n_user_frames`; the reply region starts there) and the EXACT fed input
    codes [T,8] (clip + roomtone, trimmed to T) so teacher-forced extraction
    replays generation bit-for-bit."""
    user_pcm, urms, user_end, room = _prep_user_turn(user_pcm, rms_threshold)
    ext_frames = int(max_reply_s * FRAME_HZ) + 2
    reps = int(math.ceil(ext_frames * FRAME_SAMPLES / max(1, len(room))))
    tail_pcm = np.tile(room, reps)[: ext_frames * FRAME_SAMPLES]

    all_codes = mimi_encode_codes(mimi, np.concatenate([user_pcm, tail_pcm]), device)
    T_max = int(all_codes.shape[0])
    min_frames = int(min_reply_s * FRAME_HZ)
    stop_frames = max(1, int(stop_silence_s * FRAME_HZ))
    codes, text_tokens, pcm_chunks = [], [], []
    rms = []
    speaking = False
    silent_run = 0
    with mimi.streaming(1):
        for t in range(T_max):
            out = _step_force(
                lm_gen,
                input_tokens=all_codes[t].view(1, N_AUDIO_CODEBOOKS, 1).to(device))
            tokens = out[0]                              # [1, 17, 1]: text | agent 8 | user echo 8
            text_tokens.append(int(tokens[0, 0, 0].item()))
            codes.append(tokens[0, 1:1 + N_AUDIO_CODEBOOKS, 0].long().cpu())   # agent stream only
            frame = mimi.decode(tokens[:, 1:1 + N_AUDIO_CODEBOOKS])
            pcm_f = frame[0, 0].float().cpu().numpy()
            pcm_chunks.append(pcm_f)
            r = float(np.sqrt(np.mean(pcm_f * pcm_f)))
            rms.append(r)
            if t >= user_end:
                if r > rms_threshold:
                    speaking = True
                    silent_run = 0
                else:
                    silent_run += 1
                if speaking and silent_run >= stop_frames and (t - user_end) >= min_frames:
                    break
    T_out = len(codes)
    while T_out > user_end + 1 and rms[T_out - 1] <= rms_threshold:
        T_out -= 1
    codes = torch.stack(codes)[:T_out]
    pcm = np.concatenate(pcm_chunks)[: T_out * FRAME_SAMPLES] if pcm_chunks else np.zeros(0, np.float32)
    rms_arr = np.asarray(rms, dtype=np.float32)
    txt_arr = np.asarray(text_tokens, dtype=np.int64)
    spk = np.nonzero(rms_arr > rms_threshold)[0]
    dbg = {                                   # full-timeline QA diagnostics
        "t_max": int(T_max), "n_uf": int(len(urms)), "user_end": int(user_end),
        "spk_total": int(spk.size),
        "spk_first": int(spk[0]) if spk.size else -1,
        "spk_last": int(spk[-1]) if spk.size else -1,
        "max_rms": float(rms_arr.max()) if rms_arr.size else 0.0,
        "words": int((txt_arr > TEXT_PAD).sum()),
        "epad": int((txt_arr == TEXT_EOP).sum()),
        "pad": int((txt_arr == TEXT_PAD).sum()),
    }
    return {
        "codes": codes.to(torch.int16),
        "text_tokens": txt_arr[:T_out],
        "pcm": pcm.astype(np.float32),
        "rms": rms_arr[:T_out],
        "n_user_frames": user_end,
        "user_codes": all_codes[:T_out].to(torch.int16),
        "dbg": dbg,
    }


def _decode_words(sp, text_tokens, max_chars=60):
    """sentencepiece-decode the word pieces (ids > TEXT_PAD) of a generated
    text stream — QA diagnostics only ('' when the model said nothing)."""
    if sp is None:
        return ""
    try:
        ids = [int(t) for t in text_tokens if int(t) > TEXT_PAD]
        return sp.decode(ids)[:max_chars] if ids else ""
    except Exception:
        return ""


@torch.no_grad()
def qa_generate_reply(lm_gen, mimi, sp, user_pcm, prompt_state, *,
                      max_reply_s=18.0, min_reply_s=0.5, stop_silence_s=2.0,
                      max_attempts=3, device="cuda", log=None, tag=""):
    """generate_reply + QA + stochastic retries. Sampling is unseeded, so each
    attempt is an independent draw — when the first draw lands in silence a
    fresh one usually speaks. Returns (out, reject_reason); reason is None when
    QA passed. Every rejected attempt is logged with full-timeline diagnostics
    (speech frames ANYWHERE, first/last speech frame, max rms, text-stream
    histogram, decoded words) so 'model truly silent' (spk=0, words=0) is
    distinguishable from 'replied inside the clip but VAD misplaced user_end'
    (spk_first < uend) directly from the log."""
    min_frames = max(1, int(min_reply_s * FRAME_HZ))
    attempts = max(1, int(max_attempts))
    out, reason = None, "too_short"
    for attempt in range(1, attempts + 1):
        restore_streaming_state(lm_gen, mimi, prompt_state)
        out = generate_reply(lm_gen, mimi, user_pcm, max_reply_s=max_reply_s,
                             min_reply_s=min_reply_s, stop_silence_s=stop_silence_s,
                             device=device)
        T = int(out["codes"].shape[0])
        ue = int(out["n_user_frames"])
        reply_frames = T - ue
        speak_frames = int((out["rms"][ue:] > 0.006).sum())
        speak = float((out["rms"][ue:] > 0.006).mean()) if reply_frames > 0 else 0.0
        if reply_frames < min_frames:
            reason = "too_short"
        elif not (0.05 <= speak <= 0.97):
            reason = "speak_fraction"
        else:
            return out, None
        if log is not None:
            d = out["dbg"]
            log.log(f"{tag}QA {reason} (attempt {attempt}/{attempts}): "
                    f"reply={reply_frames}f speak={speak_frames}f uend={ue} T={T} | "
                    f"tmax={d['t_max']} nuf={d['n_uf']} spk={d['spk_total']}f "
                    f"first={d['spk_first']} last={d['spk_last']} "
                    f"maxrms={d['max_rms']:.4f} words={d['words']} "
                    f"epad={d['epad']} pad={d['pad']} "
                    f"txt='{_decode_words(sp, out['text_tokens'])}'")
    return out, reason


# ---------------------------------------------------------------------------
# Emotion fusion head (~2.0M params). IMPORTANT: the deploy module
# (helium_emotion.py) ships an IDENTICAL copy of this class so the live server
# needs no training-side imports. Keep the two in sync.
# ---------------------------------------------------------------------------
class EmotionFusionHead(torch.nn.Module):
    def __init__(self, helium_dim=HELIUM_DIM, mimi_dim=MIMI_LATENT_DIM,
                 pros_dim=PROSODY_DIM, d_helium=256, d_mimi=128, d_pros=32,
                 gru_hidden=256, gru_layers=2, n_classes=len(EMOTIONS),
                 p_drop=0.10, p_modality=0.15):
        super().__init__()
        import torch.nn as nn
        self.cfg = dict(helium_dim=helium_dim, mimi_dim=mimi_dim, pros_dim=pros_dim,
                        d_helium=d_helium, d_mimi=d_mimi, d_pros=d_pros,
                        gru_hidden=gru_hidden, gru_layers=gru_layers,
                        n_classes=n_classes, p_drop=p_drop, p_modality=p_modality)
        self.p_modality = float(p_modality)
        self.proj_helium = nn.Linear(helium_dim, d_helium)
        self.proj_mimi = nn.Linear(mimi_dim, d_mimi)
        self.proj_pros = nn.Linear(pros_dim, d_pros)
        d_in = d_helium + d_mimi + d_pros
        self.in_norm = nn.LayerNorm(d_in)
        self.gru = nn.GRU(d_in, gru_hidden, num_layers=gru_layers, batch_first=True,
                          dropout=p_drop if gru_layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(gru_hidden)
        self.head = nn.Linear(gru_hidden, n_classes)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _fuse(self, helium, mimi_lat, pros):
        import torch.nn.functional as F
        if self.training and self.p_modality > 0:
            if torch.rand(()) < self.p_modality:
                helium = torch.zeros_like(helium)
            if torch.rand(()) < self.p_modality:
                mimi_lat = torch.zeros_like(mimi_lat)
            if torch.rand(()) < self.p_modality:
                pros = torch.zeros_like(pros)
        h = F.relu(self.proj_helium(helium))
        m = F.relu(self.proj_mimi(mimi_lat))
        p = F.relu(self.proj_pros(pros))
        return self.in_norm(torch.cat([h, m, p], dim=-1))

    def forward(self, helium, mimi_lat, pros):
        """[B, T, *] -> logits [B, T, C] (full-sequence, used in training)."""
        x = self._fuse(helium, mimi_lat, pros)
        y, _ = self.gru(x)
        return self.head(self.out_norm(y))

    def forward_step(self, helium, mimi_lat, pros, h=None):
        """[B, *] single step -> (logits [B, C], new GRU hidden). Live path."""
        x = self._fuse(helium, mimi_lat, pros).unsqueeze(1)
        y, h = self.gru(x, h)
        return self.head(self.out_norm(y[:, 0])), h


# ---------------------------------------------------------------------------
# Hysteresis smoother (EMA + margin + min-dwell; neutral is the fall-back).
# An identical copy ships in the deploy module.
# ---------------------------------------------------------------------------
class HysteresisSmoother:
    def __init__(self, n_classes=len(EMOTIONS), ema_alpha=0.4, enter_margin=0.15,
                 min_dwell_s=1.5, fps=FRAME_HZ, neutral_id=0):
        self.n_classes = n_classes
        self.ema_alpha = float(ema_alpha)
        self.enter_margin = float(enter_margin)
        self.min_dwell_frames = max(1, int(min_dwell_s * fps))
        self.neutral_id = int(neutral_id)
        self.reset()

    def reset(self):
        self._ema = None
        self._cur = self.neutral_id
        self._dwell = 0
        self.switches = 0
        self.steps = 0

    def update(self, probs):
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        a = self.ema_alpha
        self._ema = p if self._ema is None else (1.0 - a) * self._ema + a * p
        cand = int(np.argmax(self._ema))
        switched = False
        if cand == self._cur:
            self._dwell += 1
        else:
            dwell_ok = self._dwell >= self.min_dwell_frames
            margin_ok = float(self._ema[cand]) >= float(self._ema[self._cur]) + self.enter_margin
            if dwell_ok and margin_ok:
                self._cur = cand
                self._dwell = 0
                self.switches += 1
                switched = True
            else:
                self._dwell += 1
        self.steps += 1
        return self._cur, float(self._ema[self._cur]), switched

    @property
    def current(self) -> int:
        return self._cur

    def switches_per_min(self) -> float:
        minutes = self.steps / FRAME_HZ / 60.0
        return self.switches / minutes if minutes > 0 else 0.0


# ---------------------------------------------------------------------------
# Head checkpoint IO
# ---------------------------------------------------------------------------
def save_head(head: EmotionFusionHead, path, extra: dict | None = None) -> None:
    payload = {"state_dict": head.state_dict(), "cfg": dict(head.cfg),
               "emotions": list(EMOTIONS)}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_head(path, device="cuda") -> EmotionFusionHead:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    head = EmotionFusionHead(**payload["cfg"])
    head.load_state_dict(payload["state_dict"])
    return head.to(device).eval()


# ---------------------------------------------------------------------------
# Shard writer / reader (resume-safe)
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir, prefix, clips_per_shard=256):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.n = int(clips_per_shard)
        self.buf = []
        self.index_path = self.out_dir / f"{prefix}_index.json"
        self.completed = set()
        self.shard_idx = 0
        if self.index_path.exists():
            meta = json.loads(self.index_path.read_text())
            self.completed = set(meta.get("completed", []))
            self.shard_idx = int(meta.get("next_shard", 0))

    def add(self, rec: dict) -> bool:
        cid = rec["clip_id"]
        if cid in self.completed:
            return False
        self.buf.append(rec)
        self.completed.add(cid)
        if len(self.buf) >= self.n:
            self._flush()
        return True

    def _flush(self):
        if not self.buf:
            return
        path = self.out_dir / f"{self.prefix}_shard_{self.shard_idx:04d}.pt"
        torch.save(self.buf, path)
        self.buf = []
        self.shard_idx += 1
        self._write_index()

    def _write_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"completed": sorted(self.completed),
                                   "next_shard": self.shard_idx}))
        tmp.replace(self.index_path)

    def close(self):
        self._flush()


def iter_shard_records(feat_dir, prefix):
    """Yields records from `{prefix}_shard_*.pt` AND per-GPU worker shards
    (`{prefix}_g0_shard_*.pt`, ...), so multi-GPU output is transparent."""
    for path in sorted(Path(feat_dir).glob(f"{prefix}*_shard_*.pt")):
        for rec in torch.load(path, map_location="cpu", weights_only=False):
            yield rec


def completed_clip_ids(feat_dir, prefix) -> set:
    """Union of `completed` clip ids across all `{prefix}*_index.json` files
    (single-GPU writer + every per-GPU worker). Lets a worker skip clips that
    siblings already finished (e.g. when resuming with a different GPU count)."""
    done = set()
    for idx in Path(feat_dir).glob(f"{prefix}*_index.json"):
        try:
            done.update(json.loads(idx.read_text()).get("completed", []))
        except Exception:
            pass
    return done


def scan_frozen_shards(feat_dir, prefix, key="hidden", std_eps=1e-4) -> dict:
    """Detect shards whose per-frame features are frozen over time — the
    signature of the multi-GPU wrong-device CUDA-graph bug (workers on
    cuda:1..N replayed empty graphs, emitting the SAME hidden vector at every
    frame of every clip). A record counts as frozen when its `key` tensor has
    ~zero variation across frames. Returns {shard_name: (n_records,
    n_frozen)}. Shards with n_frozen > 0 must be deleted and re-extracted
    (extraction is resume-safe: re-running only re-does the missing clips)."""
    report = {}
    for path in sorted(Path(feat_dir).glob(f"{prefix}*_shard_*.pt")):
        n = frozen = 0
        for rec in torch.load(path, map_location="cpu", weights_only=False):
            h = rec.get(key)
            n += 1
            if isinstance(h, torch.Tensor) and h.ndim >= 2 and \
                    float(h.float().std(dim=0).mean()) < std_eps:
                frozen += 1
        report[path.name] = (n, frozen)
    return report


def read_channel(path, channel: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    ch = data[:, min(int(channel), data.shape[1] - 1)]
    if file_sr != sr:
        import torchaudio
        ch = torchaudio.functional.resample(
            torch.from_numpy(ch), file_sr, sr).numpy()
    return np.ascontiguousarray(ch, np.float32)


# ---------------------------------------------------------------------------
# Training / evaluation helpers (used by notebooks 03 and 05)
# ---------------------------------------------------------------------------
def evaluate_head(model, loader, device="cuda"):
    """Returns (logits [N, T_max, C], labels [N], masks [N, T_max]). Batches are
    zero-padded to the longest sequence in the WHOLE loader before cat — a
    plain torch.cat crashes whenever batches have different padded lengths,
    and the padded frames are always masked out downstream."""
    model.eval()
    logits_all, labels_all, masks_all = [], [], []
    t_max = 0
    with torch.no_grad():
        for H, M, PR, MK, LB in loader:
            with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
                logits = model(H.to(device), M.to(device), PR.to(device))
            t_max = max(t_max, int(logits.shape[1]))
            logits_all.append(logits.float().cpu())
            labels_all.append(LB)
            masks_all.append(MK)

    def _pad_cat(xs):
        out = []
        for x in xs:
            if x.shape[1] < t_max:
                pad = torch.zeros(x.shape[0], t_max - x.shape[1], *x.shape[2:],
                                  dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            out.append(x)
        return torch.cat(out)

    return _pad_cat(logits_all), torch.cat(labels_all), _pad_cat(masks_all)


def macro_f1_from_logits(logits, labels, masks) -> float:
    from sklearn.metrics import f1_score
    pred = logits.argmax(-1)
    yt, yp = [], []
    for i in range(logits.shape[0]):
        m = masks[i]
        if m.any():
            yt += labels[i].expand(logits.shape[1])[m].tolist()
            yp += pred[i][m].tolist()
    return float(f1_score(yt, yp, average="macro")) if yt else 0.0


def train_head(model, train_loader, val_loader, *, device="cuda", epochs=12,
               lr=3e-4, weight_decay=1e-4, patience=3, label_smoothing=0.05,
               pooled_loss_w=0.5, warmup_frac=0.05, log_fn=print):
    """Per-step CE on agent frames + turn-pooled CE. Cosine LR with warmup,
    early stopping on val macro-F1. Returns (best_f1, history)."""
    import torch.nn.functional as F
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_total = max(1, epochs * len(train_loader))
    warm = max(1, int(warmup_frac * steps_total))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(1, steps_total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    counts = torch.zeros(len(EMOTIONS))
    for _, _, _, _, LB in train_loader:
        counts += torch.bincount(LB, minlength=len(EMOTIONS))
    w = (counts.sum() / (counts + 1.0)).clamp(max=10.0)
    w = (w / w.mean()).to(device)
    log_fn(f"[train] class weights: {[round(float(x), 3) for x in w]}")
    best_f1, best_state, bad = -1.0, None, 0
    history = []
    for ep in range(int(epochs)):
        model.train()
        tot, nb = 0.0, 0
        for H, M, PR, MK, LB in train_loader:
            H, M, PR = H.to(device), M.to(device), PR.to(device)
            MK, LB = MK.to(device), LB.to(device)
            with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
                logits = model(H, M, PR)
            logits = logits.float()
            step_tgt = LB.unsqueeze(1).expand(-1, logits.shape[1])
            if MK.any():
                l_step = F.cross_entropy(logits[MK], step_tgt[MK], weight=w,
                                         label_smoothing=label_smoothing)
            else:
                l_step = logits.sum() * 0.0
            pooled = []
            for i in range(logits.shape[0]):
                m = MK[i]
                pooled.append(logits[i][m].mean(0) if m.any() else logits[i].mean(0))
            l_pool = F.cross_entropy(torch.stack(pooled), LB, weight=w,
                                     label_smoothing=label_smoothing)
            loss = l_step + float(pooled_loss_w) * l_pool
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        logits, labels, masks = evaluate_head(model, val_loader, device)
        f1 = macro_f1_from_logits(logits, labels, masks)
        history.append(dict(epoch=ep, loss=tot / max(1, nb), val_macro_f1=f1))
        log_fn(f"[train] epoch {ep}: loss={tot / max(1, nb):.4f} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1, history


# ---------------------------------------------------------------------------
# Multi-GPU workers (one spawned process per GPU, one model instance each).
# Jobs are partitioned round-robin (jobs[gpu_id::n_gpus]); each worker writes
# its own shard prefix (e.g. phaseA_g2) + log file, so resume is per-worker and
# iter_shard_records picks every worker's shards up transparently.
# ---------------------------------------------------------------------------
def _bind_worker_device(gpu_id) -> None:
    """MUST be the first CUDA-touching statement of every spawned GPU worker.
    moshi's CUDAGraphed (compile.py) captures on torch's *current* device's
    stream; a worker that keeps the inherited current device (cuda:0) while
    its tensors live on cuda:<gpu_id> captures EMPTY graphs whose replay()
    returns frozen, input-independent outputs — the multi-GPU silent-output
    bug where every sample on a GPU yields the same maxrms and 0 words.
    No-op when CUDA is unavailable (CPU smoke harness)."""
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(int(gpu_id))
    assert torch.cuda.current_device() == int(gpu_id), \
        f"failed to bind worker to cuda:{gpu_id}"


@torch.no_grad()
def assert_graphs_live(lm_gen, mimi, prompt_state, device="cuda") -> float:
    """Fail-fast proof that moshi's CUDA-graphed callables actually recompute
    on THIS device (see _bind_worker_device for the bug this guards against).

    LM probe: two trials from the same restored prompt state, each stepping
    TWICE — a trial frame, then a shared frame — and comparing the second
    step's hidden. A single step cannot work: prepare_step_input feeds the
    transformer from cache position offset-1 while input_tokens are written
    at offset+delay (lm.py L744-800), so a user frame only influences
    transformer_out one step later. Mimi probe: a 4-frame decode in a single
    streaming context, so the graphed decoder reaches replay(). Raises
    RuntimeError if either output stops depending on its input. Restores the
    prompt state before returning the LM hidden cosine (for logging)."""
    t = np.arange(FRAME_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    signals = (np.zeros(FRAME_SAMPLES, np.float32),
               0.2 * np.sin(2 * np.pi * 440 * t),
               0.2 * np.sin(2 * np.pi * 880 * t))
    codes = [mimi_encode_codes(mimi, s, device)[0] for s in signals]
    assert not (torch.equal(codes[0], codes[1])
                or torch.equal(codes[1], codes[2])), \
        "mimi encoder returned identical codes for silence and sines"
    # trial frames: only the delay-0 user codebook (q=0, lm.py delays[9]=0)
    # reaches transformer_out by the compared (second) step, so guarantee the
    # two trials differ there by construction (mimi codebook card = 2048).
    trial_a = codes[0]
    trial_b = codes[0].clone()
    trial_b[0] = (trial_b[0] + 1) % 2048
    shared = codes[0].view(1, N_AUDIO_CODEBOOKS, 1).to(device)
    hid = []
    for frame in (trial_a, trial_b):
        restore_streaming_state(lm_gen, mimi, prompt_state)
        _step_force(lm_gen,
                    input_tokens=frame.view(1, N_AUDIO_CODEBOOKS, 1).to(device))
        out = _step_force(lm_gen, input_tokens=shared)
        hid.append(out[2][0, -1].float().reshape(-1).cpu())
    cos = float(torch.nn.functional.cosine_similarity(hid[0], hid[1], dim=0))
    pcm = []
    with mimi.streaming(1):
        for frame in codes + codes[:1]:   # 4 decodes -> graphed decoder replays
            pcm.append(mimi.decode(
                frame.view(1, N_AUDIO_CODEBOOKS, 1).to(device))[0, 0].float().cpu())
    restore_streaming_state(lm_gen, mimi, prompt_state)
    # frozen decoder graph: capture runs eagerly, replays return the
    # capture-time output verbatim -> three identical PCMs in a row.
    mimi_frozen = bool(torch.equal(pcm[1], pcm[2]) and torch.equal(pcm[2], pcm[3]))
    if cos > 0.99999 or mimi_frozen:
        raise RuntimeError(
            f"CUDA graphs on {device} replay FROZEN outputs (lm cos={cos:.6f} "
            f"mimi_frozen={mimi_frozen}): this worker is not running on its "
            f"assigned GPU (torch.cuda.set_device must run before moshi's "
            f"CUDAGraphed captures). Refusing to produce input-independent "
            f"garbage.")
    return cos


def _load_stack_on(root, device, bf16_model_st=None):
    """Load sp + mimi + LM + LMGen(hidden step) on `device` inside a worker."""
    if str(device).startswith("cuda:"):
        # second safeguard (see _bind_worker_device): moshi's CUDAGraphed
        # captures on the current device's stream, so the current device MUST
        # be the target GPU before any allocation/graph warmup happens here.
        torch.cuda.set_device(int(str(device).split(":")[1]))
    P = get_paths(root)
    ensure_moshi_importable(P)
    sp = load_sp(P)
    mimi = load_mimi(P, device)
    if bf16_model_st is None:
        lm = load_lm_bnb4(P, device, torch.bfloat16)
    else:
        lm = load_lm_bf16(bf16_model_st, device, torch.bfloat16)
    lm_gen = make_lm_gen(lm, device)
    install_hidden_step(lm_gen)
    return P, sp, mimi, lm_gen


def _prompt_state_for(lm_gen, mimi, sp, P, use_prompt: bool):
    if use_prompt:
        return apply_prompts_and_snapshot(
            lm_gen, mimi, sp, str(P.VOICE_PT),
            P.TEXT_PROMPT_FILE.read_text(encoding="utf-8", errors="replace"))
    return snapshot_streaming_state(lm_gen)


def _phaseA_worker(gpu_id, n_gpus, root, jobs, cfg):
    _bind_worker_device(gpu_id)   # MUST be first: see assert_graphs_live
    device = f"cuda:{gpu_id}"
    P, sp, mimi, lm_gen = _load_stack_on(root, device)
    prompt_state = _prompt_state_for(lm_gen, mimi, sp, P, cfg.get("use_prompt", True))
    writer = ShardWriter(P.FEAT_A, f"phaseA_g{gpu_id}", cfg.get("clips_per_shard", 256))
    log = ProgressLogger(P.FEAT_A / f"extract_phaseA_g{gpu_id}.log", echo_every=10)
    done_all = completed_clip_ids(P.FEAT_A, "phaseA")
    todo = [r for r in jobs[gpu_id::n_gpus] if r["clip_id"] not in done_all]
    total = len(done_all & {r["clip_id"] for r in jobs[gpu_id::n_gpus]}) + len(todo)
    log.log(f"=== gpu{gpu_id} phaseA START | completed={total - len(todo)} "
            f"todo={len(todo)} | device={device} use_prompt={cfg.get('use_prompt', True)}",
            force=True)
    cos = assert_graphs_live(lm_gen, mimi, prompt_state, device)
    log.log(f"[gpu{gpu_id}] graphs live OK (lm cos={cos:.4f})", force=True)
    t0 = time.time()
    failed, times = 0, []
    for row in todo:
        ct0 = time.time()
        try:
            agent = read_channel(row["wav_path"], 0)
            user = read_channel(row["wav_path"], 1)
            n_frames = max(len(agent), len(user)) // FRAME_SAMPLES + 1
            stream = build_agent_text_stream(row["agent_words"], n_frames, sp)
            restore_streaming_state(lm_gen, mimi, prompt_state)
            feats = extract_clip_features(lm_gen, mimi, agent, user, stream, device)
            T = feats["hidden"].shape[0]
            mask = np.zeros(T, dtype=bool)
            for s, e in row["agent_turn_frames"]:
                mask[max(0, s):min(T, e)] = True
            ue = row.get("user_emotion")
            writer.add({
                "clip_id": row["clip_id"], **feats,
                "agent_mask": torch.from_numpy(mask),
                "label": EMO2ID[row["emotion"]],
                "user_emotion": EMO2ID.get(str(ue).lower(), -1) if ue else -1,
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
        done_n = total - len(todo) + len(times)
        remaining = max(0, total - done_n)
        avg = sum(times) / len(times)
        log.log(f"[gpu{gpu_id}] clip {done_n}/{total} | +{dt:.2f}s avg={avg:.2f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| failed={failed} | {row['clip_id']} ({row['emotion']})")
    writer.close()
    log.log(f"=== gpu{gpu_id} phaseA END | failed={failed} "
            f"| elapsed={(time.time() - t0) / 3600:.2f}h", force=True)


def phaseA_extract_multi(jobs, root, n_gpus, cfg):
    """Spawn one Phase-A extraction worker per GPU. `jobs`: list of small dicts
    (clip_id, wav_path, emotion, user_id, split, user_emotion,
    agent_turn_frames, agent_words). Blocks until all workers finish.
    Workers are capped at the job count so a tiny (smoke) job list never
    spawns idle workers that load a full model for nothing."""
    import torch.multiprocessing as tmp
    n_gpus = max(1, min(int(n_gpus), max(1, len(jobs))))
    tmp.spawn(_phaseA_worker, args=(n_gpus, root, jobs, cfg), nprocs=n_gpus, join=True)


def _gen_worker(gpu_id, n_gpus, root, samples, cfg):
    _bind_worker_device(gpu_id)   # MUST be first: see assert_graphs_live
    device = f"cuda:{gpu_id}"
    P, sp, mimi, lm_gen = _load_stack_on(root, device, bf16_model_st=cfg["model_st"])
    base_state = snapshot_streaming_state(lm_gen)
    prompt_states = {}
    for e in EMOTIONS:
        restore_streaming_state(lm_gen, mimi, base_state)
        prompt_states[e] = apply_prompts_and_snapshot(
            lm_gen, mimi, sp, str(P.VOICE_PT), f"Answer with a {e} tone.")
    writer = ShardWriter(P.GEN, f"gen_g{gpu_id}", cfg.get("clips_per_shard", 128))
    log = ProgressLogger(P.GEN / f"generate_g{gpu_id}.log", echo_every=10)
    done_all = completed_clip_ids(P.GEN, "gen")
    todo = [s for s in samples[gpu_id::n_gpus] if s["sample_id"] not in done_all]
    total = len(done_all & {s["sample_id"] for s in samples[gpu_id::n_gpus]}) + len(todo)
    log.log(f"=== gpu{gpu_id} generation START | completed={total - len(todo)} "
            f"todo={len(todo)} | device={device} "
            f"| max_attempts={int(cfg.get('max_attempts', 3))}", force=True)
    cos = assert_graphs_live(lm_gen, mimi, prompt_states[EMOTIONS[0]], device)
    log.log(f"[gpu{gpu_id}] graphs live OK (lm cos={cos:.4f})", force=True)
    t0 = time.time()
    failed, times = 0, []
    qa_fail = {"too_short": 0, "speak_fraction": 0}
    for s in todo:
        ct0 = time.time()
        try:
            user = read_channel(s["wav_path"], 1)      # right channel = user
            out, reason = qa_generate_reply(
                lm_gen, mimi, sp, user, prompt_states[s["emotion"]],
                max_reply_s=cfg.get("max_reply_s", 18.0),
                min_reply_s=cfg.get("min_reply_s", 0.5),
                stop_silence_s=cfg.get("stop_silence_s", 2.0),
                max_attempts=cfg.get("max_attempts", 3),
                device=device, log=log,
                tag=f"[gpu{gpu_id}] {s['sample_id']} ({s['emotion']}) ")
            if reason is not None:
                qa_fail[reason] += 1
                continue
            writer.add(dict(
                clip_id=s["sample_id"], src_clip=s["clip_id"], user_id=s["user_id"],
                emotion=s["emotion"], label=EMO2ID[s["emotion"]],
                codes=out["codes"],
                text_tokens=torch.from_numpy(out["text_tokens"]).to(torch.int32),
                pcm=torch.from_numpy(out["pcm"]).to(torch.float16),
                rms=torch.from_numpy(out["rms"]).to(torch.float16),
                n_user_frames=out["n_user_frames"], user_codes=out["user_codes"]))
        except Exception as e:
            failed += 1
            log.log(f"ERROR sample={s['sample_id']} ({s['emotion']}): "
                    f"{type(e).__name__}: {e} | skipped (resume-safe)", force=True)
            continue
        dt = time.time() - ct0
        times.append(dt)
        done_n = total - len(todo) + len(times)
        remaining = max(0, total - done_n)
        avg = sum(times) / len(times)
        log.log(f"[gpu{gpu_id}] sample {done_n}/{total} | +{dt:.1f}s avg={avg:.1f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| qa_rej={sum(qa_fail.values())} failed={failed} "
                f"| {s['sample_id']} ({s['emotion']})")
    writer.close()
    log.log(f"=== gpu{gpu_id} generation END | QA rejected={qa_fail} failed={failed} "
            f"| elapsed={(time.time() - t0) / 3600:.2f}h", force=True)


def generate_multi(samples, root, n_gpus, cfg):
    """Spawn one in-domain generation worker per GPU (emotionplex-7b bf16 on
    each). cfg: model_st, max_reply_s, min_reply_s, stop_silence_s,
    max_attempts, clips_per_shard. Blocks until all workers finish. Workers
    are capped at the sample count so a tiny (smoke) list never spawns idle
    workers."""
    import torch.multiprocessing as tmp
    n_gpus = max(1, min(int(n_gpus), max(1, len(samples))))
    if n_gpus > 1 and str(cfg.get("model_st", "")).endswith(".safetensors"):
        # convert safetensors -> .pt ONCE here in the parent: the repo's
        # safetensors load puts the whole state dict on GPU 0 (OOM when N
        # workers load at once) and concurrent conversions corrupt the file.
        ensure_pt_checkpoint(cfg["model_st"])
    tmp.spawn(_gen_worker, args=(n_gpus, root, samples, cfg), nprocs=n_gpus, join=True)


def _phaseB_worker(gpu_id, n_gpus, root, cfg):
    _bind_worker_device(gpu_id)   # MUST be first: see assert_graphs_live
    device = f"cuda:{gpu_id}"
    P, sp, mimi, lm_gen = _load_stack_on(root, device)
    prompt_state = _prompt_state_for(lm_gen, mimi, sp, P, True)
    import pandas as pd
    man = pd.read_parquet(P.MANIFEST)
    user_of = dict(zip(man.clip_id, man.user_id))
    holdout_users = set(man.loc[man.split == "holdout", "user_id"].unique())
    gen_recs = list(iter_shard_records(P.GEN, "gen"))
    n_test = max(1, len(gen_recs) // 10)
    test_ids = set(r["clip_id"] for r in gen_recs[-n_test:])
    writer = ShardWriter(P.FEAT_B, f"phaseB_g{gpu_id}", cfg.get("clips_per_shard", 128))
    log = ProgressLogger(P.FEAT_B / f"extract_phaseB_g{gpu_id}.log", echo_every=10)
    done_all = completed_clip_ids(P.FEAT_B, "phaseB")
    todo = [r for r in gen_recs[gpu_id::n_gpus] if r["clip_id"] not in done_all]
    total = len(done_all & {r["clip_id"] for r in gen_recs[gpu_id::n_gpus]}) + len(todo)
    log.log(f"=== gpu{gpu_id} phaseB START | completed={total - len(todo)} "
            f"todo={len(todo)} | device={device}", force=True)
    cos = assert_graphs_live(lm_gen, mimi, prompt_state, device)
    log.log(f"[gpu{gpu_id}] graphs live OK (lm cos={cos:.4f})", force=True)
    t0 = time.time()
    failed, times = 0, []
    for r in todo:
        ct0 = time.time()
        try:
            restore_streaming_state(lm_gen, mimi, prompt_state)
            feats = extract_from_codes(lm_gen, mimi, r["user_codes"].long(),
                                       r["codes"].long(),
                                       r["text_tokens"].long().numpy(), device)
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
            feats["pros"] = torch.from_numpy(prosody_features(r["pcm"].float().numpy()))
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
        done_n = total - len(todo) + len(times)
        remaining = max(0, total - done_n)
        avg = sum(times) / len(times)
        log.log(f"[gpu{gpu_id}] sample {done_n}/{total} | +{dt:.2f}s avg={avg:.2f}s "
                f"| left={remaining} ETA={avg * remaining / 3600:.2f}h "
                f"| failed={failed} | {r['clip_id']}")
    writer.close()
    log.log(f"=== gpu{gpu_id} phaseB END | failed={failed} "
            f"| elapsed={(time.time() - t0) / 3600:.2f}h", force=True)


def phaseB_extract_multi(root, n_gpus, cfg):
    """Spawn one Phase-B extraction worker per GPU. Each worker reloads the gen
    shards + manifest (deterministic split assignment) and partitions
    round-robin. Blocks until all workers finish. Workers are capped at the
    generated-sample count (read cheaply from the shard indexes) so a tiny
    (smoke) run never spawns idle workers."""
    import torch.multiprocessing as tmp
    n_samples = len(completed_clip_ids(get_paths(root).GEN, "gen"))
    n_gpus = max(1, min(int(n_gpus), max(1, n_samples)))
    tmp.spawn(_phaseB_worker, args=(n_gpus, root, cfg), nprocs=n_gpus, join=True)

