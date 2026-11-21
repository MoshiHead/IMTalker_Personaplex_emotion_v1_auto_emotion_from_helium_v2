"""helium_emotion.py — drop-in replacement for auto_emotion.py (user-voice SER).

Predicts the AGENT's emotion from PersonaPlex's own signals, per 80ms step:
  * Helium final-layer hidden state  [1, 4096]  (transformer_out, fp32)
  * Mimi reply codes                 [1, 8, 1]  (latent via stateless RRVQ decode)
  * reply PCM prosody                (1920-sample frame DSP features)

Public surface mirrors auto_emotion.AutoEmotionManager so the server can swap
managers without further changes:
  __init__(args), start(), feed(pcm_i16, input_sr)  [no-op shim],
  feed_step(hidden, reply_codes, reply_pcm, mimi=None),
  feed_chunk(hidden_chunk, codes_chunk, pcm_chunk, mimi=None),
  current_emotion_id(default=0), current_emotion(), confidence(),
  reset_session(), stop()
plus module singletons init_helium_emotion_manager(args, ...) /
get_helium_emotion_manager().

Classes EmotionFusionHead / HysteresisSmoother / prosody_features are
IDENTICAL copies of the training-side definitions (eh_common.py) so this file
has no training dependencies.

Live-vs-training parity rules (HELIUM_EMOTION_PATCH_V2):
  * The GRU sees EVERY 80ms step (training clips were whole sequences), but
    the smoother is only advanced while the assistant is actually speaking
    (reply-frame RMS >= helium_gate_rms, the same 0.006 threshold used by the
    avatar speech gate and the Phase-B training mask). Emotion cannot switch
    during silence / user speech.
  * The GRU context is reset after helium_gru_reset_silence_s of continuous
    assistant silence (a turn boundary; Phase B started every clip from a
    fresh state). Set it <= 0 to disable.
  * Smoother defaults come from config.json next to the head (written by
    training notebook 05); explicit CLI values > 0 override them.
  * Every smoother update is appended to emotion_logs/helium_emotion_*.csv
    (IMTALKER_EMOTION_LOG_DIR override) for offline inspection.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24000
FRAME_HZ = 12.5
FRAME_SAMPLES = 1920
HELIUM_DIM = 4096
MIMI_LATENT_DIM = 512
PROSODY_DIM = 6
EMOTIONS = ["neutral", "happy", "sad", "angry"]
EMOTION_NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}


# ---------------------------------------------------------------------------
# DSP prosody — identical to eh_common.prosody_features
# ---------------------------------------------------------------------------
def prosody_features(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n_frames = max(1, math.ceil(pcm.shape[0] / FRAME_SAMPLES))
    pad = n_frames * FRAME_SAMPLES - pcm.shape[0]
    if pad:
        pcm = np.pad(pcm, (0, pad))
    x = pcm.reshape(n_frames, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
    log_rms = np.log(rms + 1e-8)
    d_rms = np.concatenate([[0.0], np.diff(log_rms)]).astype(np.float32)
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
# Fusion head — identical architecture/param names to eh_common.EmotionFusionHead
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

    def _fuse(self, helium, mimi_lat, pros):
        import torch.nn.functional as F
        h = F.relu(self.proj_helium(helium))
        m = F.relu(self.proj_mimi(mimi_lat))
        p = F.relu(self.proj_pros(pros))
        return self.in_norm(torch.cat([h, m, p], dim=-1))

    def forward(self, helium, mimi_lat, pros):
        x = self._fuse(helium, mimi_lat, pros)
        y, _ = self.gru(x)
        return self.head(self.out_norm(y))

    def forward_step(self, helium, mimi_lat, pros, h=None):
        x = self._fuse(helium, mimi_lat, pros).unsqueeze(1)
        y, h = self.gru(x, h)
        return self.head(self.out_norm(y[:, 0])), h


# ---------------------------------------------------------------------------
# Hysteresis smoother — identical to eh_common.HysteresisSmoother
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


# ---------------------------------------------------------------------------
# Per-step runner (holds GRU state across steps)
# ---------------------------------------------------------------------------
class EmotionHeadRunner:
    def __init__(self, head_path, device="cuda", temperature: float = 1.0):
        payload = torch.load(str(head_path), map_location="cpu", weights_only=False)
        cfg = dict(payload["cfg"])
        self.head = EmotionFusionHead(**cfg).to(device).eval()
        self.head.load_state_dict(payload["state_dict"])
        self.device = device
        self.temperature = float(temperature or 1.0)
        self._h = None
        self.steps = 0

    def reset(self):
        self._h = None
        self.steps = 0

    @torch.no_grad()
    def update_step(self, helium_hidden, reply_codes=None, reply_pcm=None, mimi=None):
        """helium_hidden: tensor [1, 4096] or [4096] fp32 (cuda or cpu).
        reply_codes: optional tensor [1, 8, 1] / [8] int codes for this frame.
        reply_pcm:   optional float32 ndarray [1920] for this frame.
        mimi:        optional live Mimi model (for the stateless latent decode).
        Returns probs [4] float32 ndarray."""
        h = torch.as_tensor(helium_hidden, dtype=torch.float32).reshape(1, -1).to(self.device)
        if reply_codes is not None and mimi is not None:
            codes = torch.as_tensor(reply_codes).long().reshape(1, 8, -1).to(self.device)
            lat = mimi.quantizer.decode(codes)          # [1, 512, T] stateless
            m = lat[0, :, -1].reshape(1, -1).float()
        else:
            m = torch.zeros(1, MIMI_LATENT_DIM, device=self.device)
        if reply_pcm is not None:
            p = torch.from_numpy(
                prosody_features(np.asarray(reply_pcm, np.float32))[-1:]).to(self.device)
        else:
            p = torch.zeros(1, PROSODY_DIM, device=self.device)
        logits, self._h = self.head.forward_step(h, m, p, self._h)
        probs = torch.softmax(logits.float() / self.temperature, dim=-1)[0]
        self.steps += 1
        return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Manager (AutoEmotionManager-compatible)
# ---------------------------------------------------------------------------
class HeliumEmotionManager:
    def __init__(self, args) -> None:
        self.args = args
        self.initial_emotion_id = int(getattr(args, "emotion_id", 0))
        self._emotion_id = self.initial_emotion_id
        self.head_path = str(getattr(args, "helium_head_path", "") or "helium_emotion_head_final.pt")
        self.device = str(getattr(args, "helium_head_device", "") or "cuda")
        self.temperature = float(getattr(args, "helium_temperature", 1.0))
        # Smoother defaults: explicit CLI value > 0 wins, else config.json next
        # to the head (training notebook 05 grid-search), else hardcoded.
        smooth_cfg = {}
        cfg_path = Path(self.head_path).with_name("config.json")
        if cfg_path.exists():
            try:
                smooth_cfg = json.loads(cfg_path.read_text()).get("smoother", {}) or {}
            except Exception:
                smooth_cfg = {}

        def _pick(arg_name, cfg_key, hard_default):
            v = float(getattr(args, arg_name, -1.0) or -1.0)
            if v > 0:
                return v
            return float(smooth_cfg.get(cfg_key, hard_default))

        self.smoother = HysteresisSmoother(
            ema_alpha=_pick("helium_ema_alpha", "ema_alpha", 0.4),
            enter_margin=_pick("helium_enter_margin", "enter_margin", 0.15),
            min_dwell_s=_pick("helium_min_dwell_s", "min_dwell_s", 1.5),
        )
        # Smoother gating: only advance while the assistant is speaking
        # (same threshold as the avatar speech gate / Phase-B training mask).
        self.gate_rms = float(getattr(args, "helium_gate_rms", 0.006) or 0.006)
        # GRU context reset after this much continuous assistant silence
        # (turn boundary; <= 0 disables).
        self.gru_reset_silence_s = float(getattr(args, "helium_gru_reset_silence_s", 3.0) or 0.0)
        self._gru_reset_frames = max(0, int(self.gru_reset_silence_s * FRAME_HZ))
        self._silent_frames = 0
        self._lock = threading.Lock()
        self._runner: EmotionHeadRunner | None = None
        self._mimi = None
        self.steps_fed = 0
        self.gated_steps = 0
        self.feed_errors = 0
        self._conf = 0.0
        self._csv_file = None
        self._csv_writer = None
        self._log_path = None
        self._rows_since_flush = 0

    def start(self) -> None:
        t0 = time.perf_counter()
        if not Path(self.head_path).exists():
            raise FileNotFoundError(f"helium head not found: {self.head_path}")
        self._runner = EmotionHeadRunner(self.head_path, device=self.device,
                                         temperature=self.temperature)
        try:
            log_dir = os.environ.get("IMTALKER_EMOTION_LOG_DIR", "emotion_logs")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._log_path = Path(log_dir) / f"helium_emotion_{datetime.now():%Y%m%d_%H%M%S}.csv"
            self._csv_file = open(self._log_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            if self._csv_file.tell() == 0:
                self._csv_writer.writerow(
                    ["timestamp", "step", "raw_emotion", "emotion", "confidence", "switched"])
                self._csv_file.flush()
        except Exception as exc:
            print(f"[HELIUM-EMOTION] csv log disabled ({exc!r})", flush=True)
        sm = self.smoother
        print(
            f"[HELIUM-EMOTION] enabled: head={self.head_path} device={self.device} "
            f"params={sum(p.numel() for p in self._runner.head.parameters())/1e6:.2f}M "
            f"initial_emotion={EMOTION_NAMES.get(self.initial_emotion_id, self.initial_emotion_id)} "
            f"smoother=(ema={sm.ema_alpha} margin={sm.enter_margin} dwell={sm.min_dwell_frames / FRAME_HZ:.2f}s) "
            f"gate_rms={self.gate_rms} gru_reset={self.gru_reset_silence_s}s "
            f"log={self._log_path} (loaded in {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    def attach_mimi(self, mimi) -> None:
        self._mimi = mimi

    def feed(self, pcm_i16, input_sr: int) -> None:
        """AutoEmotionManager call-site compatibility shim: the helium head does
        NOT listen to the user. Mic audio is ignored by design."""
        return


    def feed_step(self, helium_hidden, reply_codes=None, reply_pcm=None, mimi=None) -> None:
        """Called once per 80ms moshi step from the reply path. Never raises."""
        try:
            if self._runner is None:
                return
            probs = self._runner.update_step(
                helium_hidden, reply_codes, reply_pcm, mimi or self._mimi)
            rms = 0.0
            if reply_pcm is not None:
                arr = np.asarray(reply_pcm, dtype=np.float32).reshape(-1)
                if arr.size:
                    rms = float(np.sqrt(np.mean(arr * arr)))
            with self._lock:
                self.steps_fed += 1
                if rms < self.gate_rms:
                    # Assistant silent: the GRU still consumed this step (as in
                    # training), but the smoother does not advance, so the
                    # emotion can never switch during silence / user speech.
                    self._silent_frames += 1
                    if 0 < self._gru_reset_frames <= self._silent_frames:
                        # Turn boundary: drop GRU context (Phase B started each
                        # clip from a fresh state).
                        self._runner.reset()
                    return
                self._silent_frames = 0
                self.gated_steps += 1
                eid, conf, switched = self.smoother.update(probs)
                if switched:
                    print(
                        f"[HELIUM-EMOTION] {EMOTION_NAMES.get(self._emotion_id)} -> "
                        f"{EMOTION_NAMES.get(eid)} (p={conf:.2f})",
                        flush=True,
                    )
                self._emotion_id = int(eid)
                self._conf = float(conf)
                self._log_row(int(np.argmax(probs)), int(eid), float(conf), bool(switched))
        except Exception as exc:
            self.feed_errors += 1
            if self.feed_errors <= 5:
                print(f"[HELIUM-EMOTION] feed error #{self.feed_errors}: {exc!r}", flush=True)

    def _log_row(self, raw_eid: int, eid: int, conf: float, switched: bool) -> None:
        w = self._csv_writer
        if w is None:
            return
        if not switched and self.gated_steps % 25 != 0:
            return  # every switch + one heartbeat row per ~2s of speech
        try:
            w.writerow([datetime.now().isoformat(timespec="milliseconds"),
                        self.steps_fed, EMOTION_NAMES.get(raw_eid, raw_eid),
                        EMOTION_NAMES.get(eid, eid), f"{conf:.4f}", int(switched)])
            self._rows_since_flush += 1
            if switched or self._rows_since_flush >= 25:
                self._csv_file.flush()
                self._rows_since_flush = 0
        except Exception:
            pass

    def feed_chunk(self, hidden_chunk, codes_chunk=None, pcm_chunk=None, mimi=None) -> None:
        """Convenience: feed a [T, 4096] chunk (e.g. 12 steps) at once."""
        T = int(hidden_chunk.shape[0])
        for t in range(T):
            codes = codes_chunk[t] if codes_chunk is not None else None
            pcm = None
            if pcm_chunk is not None:
                pcm = pcm_chunk[t * FRAME_SAMPLES:(t + 1) * FRAME_SAMPLES]
            self.feed_step(hidden_chunk[t], codes, pcm, mimi)

    def current_emotion_id(self, default: int = 0) -> int:
        with self._lock:
            return int(self._emotion_id)

    def current_emotion(self) -> str:
        return EMOTION_NAMES.get(self.current_emotion_id(), "neutral")

    def confidence(self) -> float:
        with self._lock:
            return float(self._conf)

    def reset_session(self) -> None:
        with self._lock:
            self.smoother.reset()
            self._emotion_id = self.initial_emotion_id
            self._conf = 0.0
            self._silent_frames = 0
            if self._runner is not None:
                self._runner.reset()

    def stop(self) -> None:
        try:
            if self._csv_file is not None:
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors auto_emotion.py)
# ---------------------------------------------------------------------------
_MANAGER: HeliumEmotionManager | None = None


def init_helium_emotion_manager(args, mimi=None) -> HeliumEmotionManager | None:
    """Create and start the singleton. On ANY failure, helium mode is disabled
    and the caller keeps its previous emotion behaviour."""
    global _MANAGER
    if not bool(getattr(args, "helium_emotion", False)):
        return None
    try:
        mgr = HeliumEmotionManager(args)
        mgr.start()
        if mimi is not None:
            mgr.attach_mimi(mimi)
        _MANAGER = mgr
        return mgr
    except Exception as exc:
        print(f"[HELIUM-EMOTION] disabled ({exc!r})", flush=True)
        _MANAGER = None
        return None


def get_helium_emotion_manager() -> HeliumEmotionManager | None:
    return _MANAGER

