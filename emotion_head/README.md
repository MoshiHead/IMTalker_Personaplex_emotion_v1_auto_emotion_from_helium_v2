# Helium Emotion Head — offline pipeline (RunPod)

Trainable ~2M-param head that predicts the **agent's** emotion (neutral / happy /
sad / angry) from PersonaPlex's own signals — replacing the user-voice SER
(`AutoEmotionManager` / emotion2vec):

- **Helium hidden state** `[T, 4096]` — final-layer `transformer_out`, the same
  tensor the live server captures for the adapter
- **Mimi reply codes** → 512-dim RRVQ latent (stateless decode)
- **DSP prosody** (6 features per 80 ms frame)

Everything runs through the **live-identical step path**
(`prepare_step_input` → `graphed_main` → `process_transformer_output`), so the
features match what the deployed head will see.

## Run order (one pod, 5090/32 GB recommended)

| # | Notebook | What it does | Time |
|---|----------|--------------|------|
| 00 | `00_setup_and_downloads.ipynb` | env, bnb4 weights + moshi pkg, voices, dataset download, smoke test | ~1 h (mostly download) |
| 01 | `01_dataset_inspect_and_manifest.ipynb` | sidecar parsing/validation, grouped splits, `manifest.parquet`, interleaver cross-check | ~10 min (CPU) |
| 02 | `02_extract_features_emotionplex.ipynb` | Phase A teacher-forced feature extraction (live prompt prefix ON by default) + determinism parity check | ~2 h for 8k clips (~5 h for all 24k) |
| 03 | `03_train_and_evaluate_head_phaseA.ipynb` | GATE 1 linear probes, fusion-head training, GATE 2, leakage/ECE/switching reports | ~1 h |
| 04 | `04_generate_indomain_emotionplex7b.ipynb` | in-domain replies from EmotionPlex-7b (bf16) + VARM3 + `"Answer with a {emotion} tone."` | ~4–7 h for 4k samples |
| 05 | `05_extract_indomain_and_train_phaseB.ipynb` | Phase B extraction on base bnb4, fine-tune, smoother grid-search, export `helium_emotion_head_final.pt` + `config.json` | ~2 h |
| 06 | `06_deploy_and_live_replay.ipynb` | copy `deploy_out/helium_emotion.py` + patcher, head-compat check, manager smoke, latency/switching replay, deploy checklist | ~15 min |

Restart the kernel before 04 and before 05 (frees the previous model's VRAM).

## Prerequisites

- `HF_TOKEN` env var. Only `nvidia/personaplex-7b-v1` is gated — accept its
  license on HF first. (The bnb4 community repo ships only the quantized LM;
  Mimi, SentencePiece and VARM3 are auto-fetched from the NVIDIA repo in
  notebook 00. The dataset `frabonzi/emotionplex-duplex-multispeaker` and the
  generation model `frabonzi/emotionplex-7b` are public.)
- Upload `IMTalker/prompts/RB_Robert_System_Prompt_full.txt` → `<root>/prompts/`.
- Default root `/workspace/emotion_head` (override via `EMOTION_HEAD_ROOT` or
  `eh.get_paths("/your/root")`).
- **License**: EmotionPlex dataset audio is Higgs-generated — research /
  non-commercial use only.

## Files

```
emotion_head/
  README.md                     <- this file
  eh_common.py                  <- shared module: loaders, prompt snapshot, live
                                   step override, extraction, generation, prosody,
                                   EmotionFusionHead, HysteresisSmoother, shard IO
  build_notebooks.py            <- regenerates + AST-validates all 7 notebooks
  00..06_*.ipynb                <- the pipeline
  deploy_template/
    helium_emotion.py           <- drop-in AutoEmotionManager-compatible runtime
    integrate_helium_emotion.py <- idempotent server patcher (SER + PLAIN profiles)
```

## Key decisions (change in the notebook config cells)

- `USE_LIVE_PROMPT_PREFIX = True` (nb 02/05): Robert text prompt + VARM3 prefix
  applied once, snapshotted, restored per clip — matches live hidden-state
  distribution. VARM3 is used for *generation* only in 04.
- Defaults sized for a single overnight run: 8k extraction clips, 4k generations.
- Gates: helium-only probe macro-F1 ≥ 0.55 (chance 0.25); fusion ≥ best probe +
  0.05. `GATE_STRICT=True` turns warnings into hard asserts.
- Text-stream convention (PAD=3, EOP=0 before each word, pieces from word start):
  01 prints the EmotionPlex interleaver for cross-check; fix
  `eh_common.build_agent_text_stream` BEFORE 02 if it disagrees.

## Deployment (on the live machine)

1. Copy `deploy_out/helium_emotion.py`, `deploy_out/integrate_helium_emotion.py`,
   `models/helium_emotion_head_final.pt`, `models/config.json` next to the server.
2. `python integrate_helium_emotion.py <server_file.py>` — idempotent exact-anchor
   patcher (`HELIUM_EMOTION_PATCH_V2`), writes a `.bak_helium` backup, compiles the
   result, aborts on any anchor mismatch. Works on the SER-patched server (helium
   overrides the SER emotion; init is independent of `--auto_emotion`) and on the
   plain server (manager runs + logs; no emotion seam to override).
3. Start with `--helium_emotion --helium_head_path helium_emotion_head_final.pt`.
   Smoother defaults come from `config.json` next to the head; override with
   `--helium_ema_alpha / --helium_enter_margin / --helium_min_dwell_s` (> 0 wins).
   `--helium_gate_rms 0.006` only advances the smoother while the assistant speaks
   (live parity with the Phase-B training mask); `--helium_gru_reset_silence_s 3.0`
   resets the GRU context at turn boundaries.
4. Watch for `[HELIUM-EMOTION]` switch logs; the CSV timeline lands in
   `emotion_logs/helium_emotion_*.csv` (`IMTALKER_EMOTION_LOG_DIR` override).
   Or use `runpod_helium_emotion_streaming.ipynb` which automates all of this
   (head download, patches, `start_winner_live.sh` wiring, offline replay test).
