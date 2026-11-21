import ast
import json
from pathlib import Path

d = Path("emotion_head")
nbs = sorted(d.glob("0*.ipynb"))
assert len(nbs) == 7, nbs
for p in nbs:
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    n_code = 0
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        assert isinstance(c["source"], str)
        if c["cell_type"] != "code":
            continue
        n_code += 1
        out = []
        for l in c["source"].splitlines():
            s = l.lstrip()
            out.append(l[: len(l) - len(s)] + "pass" if s.startswith(("%", "!")) else l)
        ast.parse("\n".join(out) or "pass")
    print(f"{p.name}: {len(nb['cells'])} cells ({n_code} code) - OK")

# content spot checks: critical anchors must exist in the right notebooks
checks = {
    "00_setup_and_downloads.ipynb": ["eh.BNB4_REPO", "eh.NVIDIA_REPO", "eh.DATASET_REPO",
                                     "SMOKE TEST OK", "snapshot_download",
                                     "install_hidden_step", "_env_ok.json"],
    "01_dataset_inspect_and_manifest.ipynb": ["res_emotion", "manifest.parquet",
                                              "interleaver.py", "holdout"],
    "02_extract_features_emotionplex.ipynb": ["USE_LIVE_PROMPT_PREFIX", "PARITY OK",
                                              "cosine_similarity", "ShardWriter",
                                              "apply_prompts_and_snapshot"],
    "03_train_and_evaluate_head_phaseA.ipynb": ["GATE 1", "GATE 2", "LogisticRegression",
                                                "train_head", "switches/min",
                                                "temperature"],
    "04_generate_indomain_emotionplex7b.ipynb": ["emotionplex-7b", "generate_reply",
                                                 "Answer with a {e} tone.",
                                                 "prompt_states", "QA rejected"],
    "05_extract_indomain_and_train_phaseB.ipynb": ["extract_from_codes", "test_indomain",
                                                   "helium_emotion_head_final.pt",
                                                   "config.json", "HysteresisSmoother"],
    "06_deploy_and_live_replay.ipynb": ["helium_emotion.py", "integrate_helium_emotion.py",
                                        "HeliumEmotionManager", "DEPLOY CHECKLIST",
                                        "forward_step"],
}
for name, needles in checks.items():
    text = (d / name).read_text(encoding="utf-8")
    for n in needles:
        assert n in text, f"{name}: MISSING {n!r}"
    print(f"{name}: anchors OK ({len(needles)})")
print("ALL NOTEBOOKS VALID")
