"""_smoke_harness.py — local CPU validation of the SMOKE_TEST pipeline mode.

No GPU / HF / model weights required. It checks:
  1. run-mode resolution (run_config.json > default full; EMOTION_HEAD_SMOKE
     override) and that FULL-mode paths are byte-identical to the legacy layout
     (regression guard for the in-flight pod run), smoke paths isolated under
     <root>/_smoke/.
  2. run_profile full/smoke key parity + full values == historical hard-coded.
  3. evaluate_head batch-padding (variable-T batches must not crash).
  4. multi-GPU worker caps (tiny job lists never spawn idle workers).
  5. Mini end-to-end: synthetic manifest + phaseA/phaseB/gen shards in a tmp
     smoke root, then executes the REAL generated cells of NB02 (selection),
     NB03, NB05 (training/export) and NB06 — gpu/download-tagged cells are
     skipped — and expects SMOKE PIPELINE PASSED.

Run from anywhere:  python emotion_head/_smoke_harness.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

D = Path(__file__).resolve().parent
os.chdir(D)                                  # SHARED cells use os.getcwd()
sys.path.insert(0, str(D))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import eh_common as eh

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. run-mode resolution + path isolation
# ---------------------------------------------------------------------------
print("\n== run-mode resolution & paths ==")
tmp_full = Path(tempfile.mkdtemp(prefix="eh_full_"))
tmp = Path(tempfile.mkdtemp(prefix="eh_smoke_"))
os.environ["EMOTION_HEAD_ROOT"] = str(tmp_full)
os.environ.pop("EMOTION_HEAD_SMOKE", None)

Pf = eh.get_paths()
check("missing run_config -> FULL", not Pf.SMOKE)
check("full OUT == ROOT (legacy)", Pf.OUT == Pf.ROOT)
check("full MANIFEST legacy location", Pf.MANIFEST == Pf.DATA / "manifest.parquet")
check("full GEN legacy location", Pf.GEN == tmp_full / "data" / "indomain")
check("full FEAT_A legacy location", Pf.FEAT_A == tmp_full / "features" / "phaseA")
check("full MODELS legacy location", Pf.MODELS == tmp_full / "models")
check("full DEPLOY legacy location", Pf.DEPLOY == tmp_full / "deploy_out")

eh.write_run_mode(tmp, True)
os.environ["EMOTION_HEAD_ROOT"] = str(tmp)
Ps = eh.get_paths()
check("run_config -> SMOKE", Ps.SMOKE)
check("smoke OUT under _smoke", Ps.OUT == tmp / "_smoke")
check("smoke shares raw DATA", Ps.DATA == tmp / "data" / "emotionplex")
check("smoke shares CKPT", Ps.CKPT == tmp / "checkpoints" / "personaplex_bnb4")
check("smoke GEN redirected", Ps.GEN == tmp / "_smoke" / "data" / "indomain")
check("smoke MANIFEST redirected",
      Ps.MANIFEST == tmp / "_smoke" / "data" / "emotionplex" / "manifest.parquet")
check("smoke DEPLOY redirected", Ps.DEPLOY == tmp / "_smoke" / "deploy_out")
check("worker-view consistency (root arg)", eh.get_paths(str(Ps.ROOT)).SMOKE == Ps.SMOKE)

os.environ["EMOTION_HEAD_SMOKE"] = "0"
check("env 0 overrides config file", not eh.get_paths().SMOKE)
os.environ["EMOTION_HEAD_SMOKE"] = "1"
os.environ["EMOTION_HEAD_ROOT"] = str(tmp_full)
check("env 1 overrides missing file", eh.get_paths().SMOKE)
os.environ.pop("EMOTION_HEAD_SMOKE", None)
os.environ["EMOTION_HEAD_ROOT"] = str(tmp)

# ---------------------------------------------------------------------------
# 2. run_profile parity + historical full values
# ---------------------------------------------------------------------------
print("\n== run_profile ==")
rf, rs = eh.run_profile(eh.get_paths(tmp_full)), eh.run_profile(eh.get_paths(tmp))
check("profile keys identical", set(vars(rf)) == set(vars(rs)))
check("full nb02_max_clips=8000", rf.nb02_max_clips == 8000)
check("full nb03 16/12/3", (rf.nb03_batch_size, rf.nb03_epochs, rf.nb03_patience) == (16, 12, 3))
check("full nb04 4000/18/2", (rf.nb04_n_samples, rf.nb04_max_reply_s,
                              rf.nb04_stop_silence_s) == (4000, 18.0, 2.0))
check("full nb04 max_attempts=2", rf.nb04_max_attempts == 2)
check("full nb05 8/16", (rf.nb05_epochs, rf.nb05_batch_size) == (8, 16))
check("full nb06 24/1000", (rf.nb06_replay_clips, rf.nb06_latency_iters) == (24, 1000))
check("full downloads all audio", rf.nb00_download_all_audio is True)
check("smoke nb03 4/2/2", (rs.nb03_batch_size, rs.nb03_epochs, rs.nb03_patience) == (4, 2, 2))
check("smoke nb04 12/8/1", (rs.nb04_n_samples, rs.nb04_max_reply_s,
                            rs.nb04_stop_silence_s) == (12, 8.0, 1.0))
check("smoke nb04 max_attempts=2", rs.nb04_max_attempts == 2)
check("smoke subset 10+3/emo", (rs.smoke_train_clips_per_emotion,
                                rs.smoke_val_clips_per_emotion) == (10, 3))

# ---------------------------------------------------------------------------
# 2b. ensure_smoke_subset fallback (NB01 path when NB00 smoke cells not run)
# ---------------------------------------------------------------------------
print("\n== ensure_smoke_subset ==")
_sub_file = Ps.OUT / "smoke_subset.json"
if _sub_file.exists():
    _sub_file.unlink()
_rows = []
for i in range(60):
    e, ue = eh.EMOTIONS[i % 4], eh.EMOTIONS[(i + 1) % 4]
    _rows.append(dict(path=f"stereo_wav/000/{ue}_{e}_{i:03d}.wav", res_emotion=e,
                      _manifest=("val.jsonl" if i >= 48 else "train.jsonl")))
_rows.append(dict(id="happy_happy_900", res_emotion="happy",
                  _manifest="dialogues.jsonl"))   # metadata manifest: no path
keep = eh.ensure_smoke_subset(Ps, rs, _rows)
check("fallback picks 10 train + 3 val per emotion", len(keep) == 4 * 10 + 4 * 3)
check("fallback excludes no-path metadata rows", "happy_happy_900" not in keep)
check("fallback writes the subset file", _sub_file.exists())
keep2 = eh.ensure_smoke_subset(Ps, rs, _rows)
check("subset read back from file (deterministic)", keep2 == keep)
_sub_file.unlink()
keep3 = eh.ensure_smoke_subset(Ps, rs, _rows)
check("rebuild is seeded-deterministic", keep3 == keep)
check("manifest_clip_stem path schema",
      eh.manifest_clip_stem(_rows[0]) == "happy_neutral_000")
check("manifest_clip_stem id schema",
      eh.manifest_clip_stem(_rows[-1]) == "happy_happy_900")
_sub_file.unlink()

# ---------------------------------------------------------------------------
# 2c. _prep_user_turn: duplex reply-region VAD + roomtone slice (NB04)
# ---------------------------------------------------------------------------
print("\n== _prep_user_turn ==")
_rngn = np.random.RandomState(0)
_rt = lambda n: (0.002 * _rngn.randn(int(n))).astype(np.float32)
_sp = lambda n: (0.15 * np.sin(2 * np.pi * 220 * np.arange(int(n)) / eh.SAMPLE_RATE)).astype(np.float32)
_pcm = np.concatenate([_rt(2 * eh.SAMPLE_RATE), _sp(2 * eh.SAMPLE_RATE), _rt(4 * eh.SAMPLE_RATE)])
_up, _ur, _ue, _room = eh._prep_user_turn(_pcm)
_exp = 4 * eh.FRAME_HZ + max(1, int(0.32 * eh.FRAME_HZ)) + 1     # speech ends ~4s + hangover
check("VAD finds user speech end ~4s", abs(_ue - _exp) <= 2)
check("VAD end < clip end (reply region exists)", _ue < int(_ur.shape[0]))
check("roomtone slice is quiet", float(np.sqrt((_room ** 2).mean())) < 0.01)
check("roomtone slice is 1s", len(_room) == int(eh.FRAME_HZ) * eh.FRAME_SAMPLES)
_ue2 = eh._prep_user_turn(_rt(3 * eh.SAMPLE_RATE))[2]
check("no speech -> end = clip end", _ue2 == int(np.ceil(3 * eh.SAMPLE_RATE / eh.FRAME_SAMPLES)))
del _rngn, _rt, _sp, _pcm, _up, _ur, _ue, _room, _exp, _ue2

# _stream8: legacy 16-wide gen codes (agent 8 + user echo 8) -> agent stream
_c16 = torch.arange(16).view(1, 16).expand(5, 16)
check("_stream8 slices legacy 16-wide codes", eh._stream8(_c16).shape == (5, 8))
check("_stream8 keeps agent stream first", int(eh._stream8(_c16)[0, 7]) == 7)
_c8 = torch.zeros(5, 8, dtype=torch.long)
check("_stream8 passes 8-wide through", eh._stream8(_c8) is _c8)
del _c16, _c8

# _pt_valid: complete torch.save zip vs truncated (multi-GPU conversion race)
_ptf = Path(tempfile.mkdtemp(prefix="eh_ptv_")) / "m.pt"
torch.save({"a": torch.zeros(3)}, _ptf)
check("_pt_valid accepts a complete checkpoint", eh._pt_valid(_ptf))
_b = _ptf.read_bytes()
_ptf.write_bytes(_b[: len(_b) // 2])          # truncated -> no central directory
check("_pt_valid rejects a truncated checkpoint", not eh._pt_valid(_ptf))
check("_pt_valid rejects a missing file", not eh._pt_valid(_ptf.with_name("nope.pt")))
del _ptf, _b

# ---------------------------------------------------------------------------
# 2d. qa_generate_reply: QA + stochastic retry logic (generate_reply mocked)
# ---------------------------------------------------------------------------
print("\n== qa_generate_reply (mocked) ==")
class _FakeSP:
    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)
check("_decode_words filters pad/epad", eh._decode_words(_FakeSP(), [3, 0, 5, 6, 3]) == "w5 w6")
check("_decode_words None sp -> ''", eh._decode_words(None, [5]) == "")
check("_decode_words all pad -> ''", eh._decode_words(_FakeSP(), [3, 0, 3]) == "")


def _fake_out(T, ue, rms_val, lo=0, hi=0):
    rms = np.zeros(T, np.float32)
    rms[lo:hi] = rms_val
    return {"codes": torch.zeros(T, 8, dtype=torch.int16),
            "text_tokens": np.full(T, eh.TEXT_PAD, dtype=np.int64),
            "pcm": np.zeros(T * eh.FRAME_SAMPLES, np.float32),
            "rms": rms, "n_user_frames": ue,
            "user_codes": torch.zeros(T, 8, dtype=torch.int16),
            "dbg": dict(t_max=T, n_uf=ue, user_end=ue,
                        spk_total=int((rms > 0.006).sum()),
                        spk_first=int(lo) if rms_val > 0.006 else -1,
                        spk_last=int(hi - 1) if rms_val > 0.006 else -1,
                        max_rms=float(rms.max()), words=0, epad=0, pad=T)}


_calls = {"n": 0}
def _fake_gen(*a, **kw):
    _calls["n"] += 1
    if _calls["n"] < 2:
        return _fake_out(11, 10, 0.0)                 # silent -> too_short
    return _fake_out(40, 10, 0.05, lo=12, hi=30)      # speaks -> QA pass

_orig_gen, _orig_restore = eh.generate_reply, eh.restore_streaming_state
eh.generate_reply, eh.restore_streaming_state = _fake_gen, lambda *a, **k: None
try:
    _out, _reason = eh.qa_generate_reply(None, None, _FakeSP(),
                                         np.zeros(1920, np.float32), {}, max_attempts=3)
    check("qa retry passes on 2nd attempt", _reason is None and _calls["n"] == 2)
    def _counting(fn):
        def _w(*a, **kw):
            _calls["n"] += 1
            return fn(*a, **kw)
        return _w
    _calls["n"] = 0
    eh.generate_reply = _counting(lambda *a, **kw: _fake_out(11, 10, 0.0))   # always silent
    _out, _reason = eh.qa_generate_reply(None, None, None,
                                         np.zeros(1920, np.float32), {}, max_attempts=3)
    check("qa gives up after max_attempts (too_short)",
          _reason == "too_short" and _calls["n"] == 3)
    _calls["n"] = 0
    eh.generate_reply = _counting(lambda *a, **kw: _fake_out(40, 10, 0.0))   # long but silent
    _out, _reason = eh.qa_generate_reply(None, None, None,
                                         np.zeros(1920, np.float32), {}, max_attempts=2)
    check("qa speak_fraction reject", _reason == "speak_fraction" and _calls["n"] == 2)
finally:
    eh.generate_reply, eh.restore_streaming_state = _orig_gen, _orig_restore
del _calls, _fake_gen, _fake_out, _FakeSP, _orig_gen, _orig_restore, _out, _reason

# ---------------------------------------------------------------------------
# 2e. multi-GPU device binding + graph liveness + frozen-shard scan
# ---------------------------------------------------------------------------
print("\n== 2e. device binding / graph liveness / frozen-shard scan ==")
import inspect as _inspect
for _fn in (eh._phaseA_worker, eh._gen_worker, eh._phaseB_worker):
    _body = [l for l in _inspect.getsource(_fn).splitlines()[1:] if l.strip()]
    check(f"{_fn.__name__} binds its CUDA device first",
          _body[0].strip().startswith("_bind_worker_device(gpu_id)"))
eh._bind_worker_device(0)   # no CUDA here -> must be a safe no-op
check("_bind_worker_device is a safe no-op without CUDA", True)

import contextlib as _cl


class _MockMimi:
    """decode: eager for calls 1-2; when frozen=True, calls 3+ replay the
    call-2 output (mirrors a CUDAGraphed decoder captured on the wrong
    device, whose replay returns the capture-time output verbatim)."""

    def __init__(self, frozen):
        self.frozen, self._n, self._cap = frozen, 0, None

    @_cl.contextmanager
    def streaming(self, n):
        yield

    def decode(self, codes):
        self._n += 1
        if self.frozen and self._n > 2:
            return self._cap
        out = torch.full((1, 1, 16), float(codes.sum()))
        if self._n == 2:
            self._cap = out
        return out


def _mock_encode_factory():
    calls = {"n": 0}

    def _enc(mimi, pcm, device):
        calls["n"] += 1
        return torch.full((1, eh.N_AUDIO_CODEBOOKS), float(calls["n"]),
                          dtype=torch.long)

    return _enc


def _mk_step(frozen):
    prev = {"v": 0.0}

    def _step(lm_gen, **kw):
        h = torch.arange(8, dtype=torch.float32)
        if not frozen:
            # lm.py pipeline delay: this step's hidden reflects LAST step's
            # input; steer one of the LAST-position components (what
            # assert_graphs_live reads) so trials are clearly separable
            h[4 + int(prev["v"]) % 4] += 100.0
        prev["v"] = float(kw["input_tokens"].sum())
        return None, None, h.view(1, 2, 4)
    return _step


_real_enc = eh.mimi_encode_codes
_real_sf, _real_rs = eh._step_force, eh.restore_streaming_state
try:
    eh.mimi_encode_codes = _mock_encode_factory()
    eh._step_force = _mk_step(frozen=False)
    eh.restore_streaming_state = lambda *a, **k: None
    cos = eh.assert_graphs_live(None, _MockMimi(False), {}, "cpu")
    check("assert_graphs_live passes on live graphs", 0 < cos < 0.99999)
    eh._step_force = _mk_step(frozen=True)
    try:
        eh.assert_graphs_live(None, _MockMimi(True), {}, "cpu")
        check("assert_graphs_live raises on frozen graphs", False)
    except RuntimeError as e:
        check("assert_graphs_live raises on frozen graphs", "FROZEN" in str(e))
finally:
    eh.mimi_encode_codes = _real_enc
    eh._step_force, eh.restore_streaming_state = _real_sf, _real_rs
del (_real_enc, _real_sf, _real_rs, _mk_step, _mock_encode_factory, _MockMimi,
     _cl, _inspect, cos)

_scan_dir = tmp / "scan"
_scan_dir.mkdir(parents=True, exist_ok=True)
_w = eh.ShardWriter(_scan_dir, "phaseA", clips_per_shard=10)
_w.add(dict(clip_id="ok", hidden=torch.randn(50, eh.HELIUM_DIM)))
_w.add(dict(clip_id="bad", hidden=torch.full((50, eh.HELIUM_DIM), 3.14)))
_w.close()
_rep = eh.scan_frozen_shards(_scan_dir, "phaseA")
check("scan_frozen_shards flags time-frozen hidden features",
      sum(v[0] for v in _rep.values()) == 2
      and sum(v[1] for v in _rep.values()) == 1)
del _scan_dir, _w, _rep

# ---------------------------------------------------------------------------
# helpers shared by sections 3 and 5
# ---------------------------------------------------------------------------
def _synth_rec(cid, label, split, uid, t, user_emo=1):
    mask = torch.zeros(t, dtype=torch.bool)
    mask[t // 3:] = True
    return dict(clip_id=cid,
                hidden=torch.randn(t, eh.HELIUM_DIM, dtype=torch.float16),
                mimi_lat=torch.randn(t, eh.MIMI_LATENT_DIM, dtype=torch.float16),
                pros=torch.randn(t, eh.PROSODY_DIM, dtype=torch.float16),
                agent_mask=mask, label=int(label), user_emotion=int(user_emo),
                user_id=str(uid), split=split,
                text_stream=torch.zeros(t, dtype=torch.int32))


class _DS(torch.utils.data.Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        return (r["hidden"], r["mimi_lat"], r["pros"], r["agent_mask"], int(r["label"]))


def _collate(batch):
    T = max(b[0].shape[0] for b in batch)
    H = torch.zeros(len(batch), T, eh.HELIUM_DIM)
    M = torch.zeros(len(batch), T, eh.MIMI_LATENT_DIM)
    PR = torch.zeros(len(batch), T, eh.PROSODY_DIM)
    MK = torch.zeros(len(batch), T, dtype=torch.bool)
    LB = torch.zeros(len(batch), dtype=torch.long)
    for i, (h, m, p, mk, lb) in enumerate(batch):
        t = h.shape[0]
        H[i, :t], M[i, :t], PR[i, :t], MK[i, :t], LB[i] = \
            h.float(), m.float(), p.float(), mk, lb
    return H, M, PR, MK, LB


# ---------------------------------------------------------------------------
# 3. evaluate_head variable-T padding
# ---------------------------------------------------------------------------
print("\n== evaluate_head padding ==")
_model = eh.EmotionFusionHead()
_var_t = [_synth_rec("v0", 0, "val", "u0", 12), _synth_rec("v1", 1, "val", "u1", 40),
          _synth_rec("v2", 2, "val", "u2", 25)]
_loader = torch.utils.data.DataLoader(_DS(_var_t), batch_size=1, collate_fn=_collate)
_lg, _lb, _mk = eh.evaluate_head(_model, _loader, "cpu")
check("evaluate_head pads to loader T_max",
      _lg.shape == (3, 40, 4) and _mk.shape == (3, 40))
check("padding is masked out", bool(_mk[0, 12:].sum() == 0))
check("macro_f1 runs on padded logits", 0.0 <= eh.macro_f1_from_logits(_lg, _lb, _mk) <= 1.0)

# ---------------------------------------------------------------------------
# 4. multi-GPU worker caps (spawn monkeypatched — nothing actually spawns)
# ---------------------------------------------------------------------------
print("\n== worker caps ==")
import torch.multiprocessing as _tmp_mod

_spawn_calls = []


class _FakeSpawn:
    def __call__(self, fn, args, nprocs, join):
        _spawn_calls.append((fn.__name__, nprocs))


_real_spawn = _tmp_mod.spawn
_tmp_mod.spawn = _FakeSpawn()
try:
    eh.phaseA_extract_multi([{"clip_id": "a"}], str(tmp), 8, {})
    eh.generate_multi([{"sample_id": "s"}, {"sample_id": "t"}], str(tmp), 8, {})
    _w = eh.ShardWriter(Ps.GEN, "gen", clips_per_shard=4)   # 3 recs -> phaseB cap 3
    for i in range(3):
        _w.add(dict(clip_id=f"gen_{i:05d}__src"))
    _w.close()
    eh.phaseB_extract_multi(str(tmp), 8, {})
finally:
    _tmp_mod.spawn = _real_spawn
check("phaseA caps at 1 job", _spawn_calls[0] == ("_phaseA_worker", 1))
check("gen caps at 2 samples", _spawn_calls[1] == ("_gen_worker", 2))
check("phaseB caps at 3 gen clips", _spawn_calls[2] == ("_phaseB_worker", 3))

# ---------------------------------------------------------------------------
# 4b. diagnose_gen_dir (NB05 guard message)
# ---------------------------------------------------------------------------
print("\n== diagnose_gen_dir ==")
tmp2 = Path(tempfile.mkdtemp(prefix="eh_diag_"))
eh.write_run_mode(tmp2, True)
P2 = eh.get_paths(tmp2)
P2.GEN.mkdir(parents=True, exist_ok=True)
(P2.GEN / "generate.log").write_text("worker start\nQA rejected={'too_short': 12} failed=0\n")
m = eh.diagnose_gen_dir(P2)
check("diagnose: empty dirs -> run NB04", "no gen shards anywhere" in m)
check("diagnose: surfaces QA log tail", "too_short" in m)
_w = eh.ShardWriter(P2.ROOT / "data" / "indomain", "gen", clips_per_shard=4)
_w.add(dict(clip_id="gen_00000__x"))
_w.close()
m2 = eh.diagnose_gen_dir(P2)
check("diagnose: detects full-mode strays", "FULL-mode gen dir" in m2 and "FRESH kernel" in m2)
del tmp2, P2, m, m2


# ---------------------------------------------------------------------------
# 5. mini end-to-end on synthetic data — executes the REAL generated cells
# ---------------------------------------------------------------------------
print("\n== mini end-to-end (NB02 selection + NB03 + NB05 + NB06, CPU) ==")
import pandas as pd

rng = np.random.RandomState(eh.SEED)

# synthetic phaseA shards: 10/emo train, 3/emo val, 1/emo holdout (smoke sizes)
wa = eh.ShardWriter(Ps.FEAT_A, "phaseA", clips_per_shard=8)
_cid = 0
for split, per_emo in (("train", 10), ("val", 3), ("holdout", 1)):
    for emo in range(4):
        for _ in range(per_emo):
            t = int(rng.randint(24, 48))
            wa.add(_synth_rec(f"clip_{_cid:03d}", emo, split, f"user{_cid % 17}", t,
                              user_emo=(emo + 1) % 4))
            _cid += 1
wa.close()

# synthetic phaseB shards: 9 train + 1 holdout + 2 test_indomain
wb = eh.ShardWriter(Ps.FEAT_B, "phaseB", clips_per_shard=8)
_b_splits = ["train"] * 9 + ["holdout"] + ["test_indomain"] * 2
for i, sp in enumerate(_b_splits):
    wb.add(_synth_rec(f"gen_{i:05d}__clip_{i:03d}", i % 4, sp, f"user{i % 17}",
                      int(rng.randint(24, 48))))
wb.close()

# synthetic manifest (needed by the artifact check + NB02 selection cell)
rows = []
_cid = 0
for split, per_emo in (("train", 10), ("val", 3), ("holdout", 1)):
    for emo in range(4):
        for _ in range(per_emo):
            e = eh.EMOTIONS[emo]
            rows.append(dict(clip_id=f"clip_{_cid:03d}",
                             rel_path=f"stereo_wav/000/{e}_{e}_{_cid:03d}.wav",
                             wav_path=f"/data/stereo_wav/000/{e}_{e}_{_cid:03d}.wav",
                             has_wav=True, manifest=("val.jsonl" if split == "val"
                                                     else "train.jsonl"),
                             duration=8.0, emotion=e, user_id=f"user{_cid % 17}",
                             user_emotion=eh.EMOTIONS[(emo + 1) % 4],
                             n_turns=2, n_agent_turns=1,
                             agent_turn_frames="[[8, 40]]", alignments="[]",
                             split=split))
            _cid += 1
Ps.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(Ps.MANIFEST, index=False)
print("synthetic data:", len(rows), "manifest rows |",
      len(list(Ps.FEAT_A.glob('phaseA*_shard_*.pt'))), "phaseA shards |",
      len(list(Ps.FEAT_B.glob('phaseB*_shard_*.pt'))), "phaseB shards")

SKIP_TAGS = ("gpu", "download")


def run_nb(name):
    nb = json.loads((D / name).read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    n_run = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        tags = set(c.get("metadata", {}).get("tags", []))
        if tags & set(SKIP_TAGS):
            print(f"  [{name}] cell {i}: skipped (tags: {sorted(tags & set(SKIP_TAGS))})")
            continue
        src = "\n".join(l for l in c["source"].splitlines()
                        if not l.lstrip().startswith(("%", "!")))
        exec(compile(src, f"{name}#cell{i}", "exec"), ns)
        n_run += 1
    print(f"  [{name}] {n_run} cells executed")
    return ns


ns02 = run_nb("02_extract_features_emotionplex.ipynb")
sel_splits = set(ns02["sel"]["split"].unique())
check("NB02 selection covers train+val+holdout",
      {"train", "val", "holdout"} <= sel_splits)

ns03 = run_nb("03_train_and_evaluate_head_phaseA.ipynb")
check("NB03 saved head_phaseA.pt", (Ps.MODELS / "head_phaseA.pt").exists())
check("NB03 wrote metrics", (Ps.REPORTS / "phaseA_metrics.json").exists())

ns05 = run_nb("05_extract_indomain_and_train_phaseB.ipynb")
check("NB05 exported final head", (Ps.MODELS / "helium_emotion_head_final.pt").exists())
_cfg = json.loads((Ps.MODELS / "config.json").read_text())
check("NB05 config has smoother + smoke marker",
      "smoother" in _cfg and _cfg.get("smoke_test") is True)

ns06 = run_nb("06_deploy_and_live_replay.ipynb")
check("NB06 deploy files copied", (Ps.DEPLOY / "helium_emotion.py").exists()
      and (Ps.DEPLOY / "integrate_helium_emotion.py").exists())

print("\n== summary ==")
if FAILURES:
    print(f"{len(FAILURES)} FAILURES:")
    for f in FAILURES:
        print("  FAIL", f)
    sys.exit(1)
print("SMOKE HARNESS: ALL CHECKS PASSED (including SMOKE PIPELINE PASSED from NB06)")

