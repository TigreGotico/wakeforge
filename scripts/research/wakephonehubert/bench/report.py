"""report.py [--bench-dir DIR]

Collects results/*/*/results.json of the benchmark into RESULTS.md.
"""
import argparse
import json
from pathlib import Path

B = Path("work/bench")
R = B / "results"
ROWS = [("logmel", "log-mel"), ("wakehubert", "WakeHuBERT-tiny"), ("wakephonehubert", "WakePhoneHuBERT"),
        ("hubert-base", "HuBERT-base (teacher)"), ("wav2vec2-espeak", "wav2vec2-xlsr-53-espeak (teacher)"),
        ("efficientat-mn10", "EfficientAT mn10_as (teacher)")]
FRAME_ONLY = {"efficientat-mn10"}


def load(task, feat):
    f = R / task / feat / "results.json"
    return json.loads(f.read_text()) if f.exists() else None


def pct(x):
    return f"{100 * x:.2f}"


def task_table(task, key, label, lower_better=False, skip=()):
    out = [f"| Featurizer | seed 0 | seed 1 | mean |", "|---|---|---|---|"]
    weights = []
    for feat, name in ROWS:
        if feat in skip:
            continue
        r = load(task, feat)
        if r is None:
            note = "pending: heads not trained yet" if feat == "wakephonehubert" else "not run yet"
            out.append(f"| {name} | {note} | | |")
            continue
        v = [r["seeds"][s][key] for s in ("0", "1") if s in r["seeds"]]
        out.append(f"| {name} | {pct(v[0])} | {pct(v[1]) if len(v) > 1 else ''} | {pct(sum(v) / len(v))} |")
        w = r["seeds"]["0"].get("weights") or r["seeds"]["0"].get("weights_mean")
        if w and len(w) > 1:
            weights.append(f"- {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in w.items()))
    s = "\n".join(out)
    if weights:
        s += f"\n\nLearned layer weights, seed 0{' (mean over the five folds)' if task == 'esc50' else ''}:\n\n" + "\n".join(weights)
    return s


def main():
    global B, R
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(B))
    B = Path(ap.parse_args().bench_dir)
    R = B / "results"
    md = ["# WakeHuBERT downstream benchmark", "",
          "SUPERB protocol: the featurizer is frozen; a learned softmax-weighted sum mixes its entries (each standardised with "
          "training-split statistics; entries narrower or wider than the featurizer's width pass through a learned linear first); "
          "a small head reads the mix. Same head and budget for every featurizer, seeds 0 and 1 both reported. One results.json "
          "per task and featurizer sits under results/.", "",
          "Entries: log-mel = the trunk's normalised log-mel with frames 2t and 2t+1 stacked (128). WakeHuBERT-tiny = stem and "
          "eight blocks (9 x 256) plus the 128 output projected to 256. WakePhoneHuBERT = those plus the vad and ipa-posterior "
          "head outputs. HuBERT-base = 13 hidden states. wav2vec2-xlsr-53-espeak = 25 hidden states. EfficientAT = its "
          "pooled 960-dim embedding, clip-level tasks only, input left-padded with zeros to at least 5 s.", ""]
    md += ["## Phoneme recognition (LibriSpeech test-clean, PER %, lower is better)", "",
           "Linear CTC on the weighted sum; targets from the wav2vec2-xlsr-53-espeak tokenizer (en-us) with pad, <s>, </s>, <unk> "
           "and | dropped. Adam lr 1e-2, batches of 160 s of audio, at most 5 epochs over train-clean-100, early stop on "
           "dev-clean PER (patience 2). Both seeds see the same batch order; the seed sets the head's initialisation. "
           "This budget is far below SUPERB's (100k steps of 32 utterances, about 110 epochs), so PER here is higher "
           "than a SUPERB run of the same featurizer would give.", "",
           task_table("pr", "test_per", "PER", skip=FRAME_ONLY | {"wav2vec2-espeak"})]
    d = R / "pr/wav2vec2-espeak-direct/results.json"
    if d.exists():
        j = json.loads(d.read_text())
        md += ["", f"wav2vec2-xlsr-53-espeak (teacher), its own CTC head, no probe training, greedy decoding: test-clean PER "
               f"{pct(j['test_per'])} (dev-clean {pct(j['dev_per'])}). This is the upper reference for the IPA output; its symbol "
               "table is the targets' table, but it was fine-tuned on Common Voice phonemisations, not on LibriSpeech. "
               "No linear probe was trained on its hidden states for this task."]
    md += ["", "## Sound classification (ESC-50, 5-fold accuracy %)", "",
           "Mean pooling, weighted sum, linear. Fold k tested, fold k+1 dev, other three train; mean over the folds. Adam lr 1e-3, "
           "batch 256, at most 50 epochs, early stop on dev accuracy (patience 5).", "", task_table("esc50", "test_acc", "acc")]
    lid = load("lid", "wakehubert") or load("lid", "logmel")
    sizes = f" Train {lid['sizes']['train']}, dev {lid['sizes']['dev']}, test {lid['sizes']['test']} clips." if lid else ""
    md += ["", "## Language identification (VoxLingua107, 10 languages, accuracy %)", "",
           "Languages: sv nl hy fr fa ar da lv fi et, the ten with the most clips in the official dev set. Training clips from "
           "the first two training shards of each language, 10% of videos held out as dev; test is the official dev set "
           f"restricted to these languages.{sizes} Same head and budget as ESC-50.", "", task_table("lid", "test_acc", "acc")]
    sid = load("sid", "wakehubert") or load("sid", "logmel")
    sizes = f" Train {sid['sizes']['train']}, dev {sid['sizes']['dev']}, test {sid['sizes']['test']} utterances." if sid else ""
    md += ["", "## Speaker identification (VoxCeleb1, 1,251 speakers, accuracy %)", "",
           "Official identification split (iden_split.txt). VoxCeleb1 is streamed from the ProgramComputer/voxceleb copy of the "
           "official zips (CC BY 4.0) and never stored; only each utterance's mean-pooled entries are kept, and mean pooling "
           f"followed by the weighted sum and a linear layer is the same model as SUPERB's SID head. SUBSET: all 1,251 speakers, "
           "training side cut to a fixed random sample of 8 utterances per speaker (numpy default_rng(0)), official dev and test "
           f"utterances kept whole, so these numbers are not comparable with SUPERB's full-split SID.{sizes} Same head and budget as ESC-50.", "",
           task_table("sid", "test_acc", "acc")]
    v = R / "vad/results.json"
    md += ["", "## Voice activity (LibriParty eval, 50 sessions, speech-frame F1 %)", ""]
    if v.exists():
        j = json.loads(v.read_text())
        md += ["| System | F1 at 0.5 | precision | recall | F1 at dev threshold |", "|---|---|---|---|---|"]
        for k in ["silero", "webrtc-0", "webrtc-1", "webrtc-2", "webrtc-3", "wakephonehubert"]:
            if k in j:
                r = j[k]; e = r["eval_at_0.5"]
                dt = f"{pct(r['eval_at_dev_threshold']['f1'])} (t={r['dev_tuned_threshold']})" if "eval_at_dev_threshold" in r else "binary output"
                name = {"silero": "Silero VAD v5", "wakephonehubert": "WakePhoneHuBERT vad head"}.get(k, f"WebRTC VAD mode {k[-1]}")
                md.append(f"| {name} | {pct(e['f1'])} | {pct(e['precision'])} | {pct(e['recall'])} | {dt} |")
        if "wakephonehubert" not in j:
            md.append("| WakePhoneHuBERT vad head | pending: heads not trained yet | | | |")
        md += ["", "The reference marks each LibriSpeech utterance's whole span as speech, including its pauses and leading and "
               "trailing silence, so a frame-accurate VAD loses recall against it. That is why Silero's precision is near 1 "
               "while its recall is about 0.7, and why its dev-tuned threshold sits at the bottom of the grid. Compare systems "
               "with each other on this reference, not with an absolute F1."]
    md += ["", "## Published reference numbers (not re-run here)", "",
           "From SUPERB (Yang et al., arXiv 2105.01051v4), Table 2, Section 4: FBANK PR 82.01 PER, SID 8.5E-4 accuracy as "
           "printed; HuBERT Base PR 5.41 PER, SID 81.42 accuracy. DistilHuBERT is not in that paper: its SUPERB numbers come "
           "from the DistilHuBERT paper (Chang et al., arXiv 2110.01900v4), Table 1: PR 16.27 PER, SID 73.54 accuracy, "
           "23.49M parameters. These used SUPERB's full budget; the rows above use a much smaller one."]
    sp = R / "speed.json"
    if sp.exists():
        j = json.loads(sp.read_text())
        names = {"logmel": "log-mel", "wakehubert": "WakeHuBERT-tiny (PyTorch)", "wakehubert-onnx-int8": "WakeHuBERT-tiny (shipped int8 ONNX)",
                 "wakephonehubert": "WakePhoneHuBERT", "hubert-base": "HuBERT-base", "wav2vec2-espeak": "wav2vec2-xlsr-53-espeak",
                 "efficientat-mn10": "EfficientAT mn10_as", "silero": "Silero VAD v5", "webrtc": "WebRTC VAD"}
        md += ["", "## Size and speed", "",
               "CPU latency per 80 ms of audio on one thread: one call per 80 ms hop, on a 2.5 s window for our models and for "
               "HuBERT and wav2vec2 (which have no streaming window), on the 5 s window used here for EfficientAT, and on the "
               "native chunk for Silero (32 ms) and WebRTC (10 ms). Median of 30 calls.", "",
               "| Model | parameters | weights file | ms per 80 ms |", "|---|---|---|---|"]
        for k, n in names.items():
            r = j.get(k)
            if not r:
                continue
            if "pending" in r:
                md.append(f"| {n} | pending | | |"); continue
            p = f"{r['params'] / 1e6:.2f} M" if r.get("params") else "n/a"
            fb = f"{r['file_bytes'] / 1e6:.1f} MB" if r.get("file_bytes") else "n/a"
            md.append(f"| {n} | {p} | {fb} | {r['ms_per_80ms']:.2f} |")
    md += ["", "## Running the WakePhoneHuBERT rows", "",
           "Once the heads directory holds vad_head.pt and ipa_head.pt, run `bash run_wph.sh`, then `python report.py` to "
           "rebuild this file."]
    (B / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
