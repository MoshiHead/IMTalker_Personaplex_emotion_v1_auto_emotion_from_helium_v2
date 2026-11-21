import os
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("IMTALKER_EMOTION_LOG_DIR", tempfile.mkdtemp(prefix="helium_compat_logs_"))

sys.path.insert(0, "emotion_head")
sys.path.insert(0, "emotion_head/deploy_template")
import eh_common as eh
import helium_emotion as hd

head = eh.EmotionFusionHead()
with tempfile.TemporaryDirectory() as td:
    p = td + "/head.pt"
    eh.save_head(head, p, extra={"val_macro_f1": 0.0})
    payload = torch.load(p, map_location="cpu", weights_only=False)
    d = hd.EmotionFusionHead(**payload["cfg"])
    d.load_state_dict(payload["state_dict"])
    r = hd.EmotionHeadRunner(p, device="cpu")
    probs = r.update_step(torch.randn(1, 4096), None,
                          (np.random.randn(1920) * 0.05).astype("float32"))
    print("runner probs:", probs.round(3), "sum:", float(probs.sum()))
    mgr_args = type("A", (), {
        "emotion_id": 0, "helium_head_path": p, "helium_head_device": "cpu",
        "helium_temperature": 1.0, "helium_ema_alpha": 0.4,
        "helium_enter_margin": 0.15, "helium_min_dwell_s": 1.5})()
    mgr = hd.HeliumEmotionManager(mgr_args)
    mgr.start()
    mgr.feed(b"\x00" * 1920, 24000)   # mic shim: no-op
    for _ in range(30):   # loud pcm -> smoother advances (above gate_rms)
        mgr.feed_step(torch.randn(1, 4096), None,
                      (np.random.randn(1920) * 0.1).astype("float32"))
    print("manager:", mgr.current_emotion_id(), mgr.current_emotion(),
          "steps:", mgr.steps_fed, "gated:", mgr.gated_steps, "errors:", mgr.feed_errors)
    for _ in range(10):   # silence -> smoother frozen, GRU reset after 3s
        mgr.feed_step(torch.randn(1, 4096), None, np.zeros(1920, dtype="float32"))
    print("after silence:", mgr.current_emotion_id(), "gated:", mgr.gated_steps,
          "gru_steps:", mgr._runner.steps)
    mgr.reset_session()
    print("after reset:", mgr.current_emotion_id(), mgr.current_emotion())
print("DEPLOY_COMPAT_OK")
