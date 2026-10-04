"""write_validation_md.py <build_dir> <out_md> <clips_manifest.json> <commands.txt>

Writes VALIDATION.md from a build's config.json and validation_report.json."""
import json
import sys
from pathlib import Path


def f(x, n=6):
    return f"{x:.{n}g}" if isinstance(x, float) else str(x)


def main():
    b, out_md, manifest, commands = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    cfg = json.loads((b / "config.json").read_text())
    r = json.loads((b / "validation_report.json").read_text())
    man = json.loads(manifest.read_text())
    L = []
    w = L.append
    w("# WakePhoneHuBERT export: validation\n")
    w("The published WakeHuBERT-tiny int8 graph (sha256 `" + cfg["trunk"]["sha256"] + "`), unchanged, with the two heads "
      "trained by `train_pool.py` attached to its taps. The heads were trained on the **float** PyTorch trunk "
      "(`Trunk(load_trunk())`, the safetensors model); in this file they read the **int8** trunk's taps.\n")
    w("## Outputs\n")
    w("| output | shape | rate |")
    w("|---|---|---|")
    for k, v in cfg["outputs"].items():
        w(f"| `{k}` | {' x '.join(v['shape'])} | {v['frame_rate_hz']:g} Hz |")
    w("")
    w("`features` layout (start, end): " + ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in cfg["output"]["layout"].items())
      + f"; {cfg['feature_dim']} values per frame.\n")
    w(f"The `ipa` output has {cfg['vocab']['size']} values, not 393: it is the wav2vec2-xlsr-53-espeak-cv-ft tokenizer's "
      f"whole id table, in which id 0 `<pad>` is the CTC blank and ids 1-3 (`<s>`, `</s>`, `<unk>`) are never targets, "
      f"leaving {cfg['vocab']['size'] - 4} phone symbols. The table is in `vocab.json`.\n")
    w("## Head precision\n")
    rule = cfg["acceptance_rule"]
    w(f"Rule, on {len(man['accept'])} held-out acceptance clips (disjoint from the {len(man['calib'])} calibration clips), "
      f"against the same build with float32 heads: vad and ipa probabilities move by less than {rule['tol']} and decisions "
      f"agree on at least {rule['agree']} of frames (vad at 0.5, ipa argmax).\n")
    w("| head | dtype tried | max abs prob diff | decision agreement | passes | full file bytes |")
    w("|---|---|---|---|---|---|")
    for t in cfg["heads_dtype_trials"].values():
        w(f"| {t['head']} | {t['dtype']} | {t['max_abs_prob_diff']:.5f} | {t['decision_agreement']:.5f} | {'yes' if t['pass'] else 'no'} | {t['full_file_bytes']:,} |")
    w("")
    w("Dtypes, smallest first: int8-static is onnxruntime static QDQ quantisation of the head's Conv nodes (int8 per-channel "
      "weights, uint8 activations calibrated with MinMax on the calibration clips run through the int8 trunk); int8-weights stores "
      "the head's Conv weights as int8 with a symmetric per-output-channel scale, dequantised to float32 at load, other head "
      "parameters float16, arithmetic float32; float16 stores every head parameter in float16, cast to float32 at load. Each head is "
      "tried alone with the other heads at float32 and keeps the smallest dtype that passes; the combined build passed the same rule.\n")
    w("Chosen: " + ", ".join(f"{h} {d}" for h, d in cfg["heads_dtype"].items()) + ".\n")
    w("## Files\n")
    w("| file | bytes |")
    w("|---|---|")
    for k, v in r["sizes_bytes"].items():
        w(f"| `{k}` | {v:,} |")
    w("")
    w("ONNX checker: " + r["checker"] + ".\n")
    w("## hubert is the published featurizer, bit for bit\n")
    w("| input | samples x batch | frames | full model, default threads | full model, 1 thread | features file slice |")
    w("|---|---|---|---|---|---|")
    for x in r["bit_identity"]:
        w(f"| {x['input']} | {x['samples']} x {x['batch']} | {x['frames']} | {x['hubert_bitwise_equal_default_threads']} | "
          f"{x['hubert_bitwise_equal_1_thread']} | {x['features_only_hubert_slice_bitwise_equal']} |")
    w(f"\nAll equal: **{r['bit_identity_all_true']}**.\n")
    clips = {k: sum(1 for c in r["real_clips"] if c["file"].startswith(k)) for k in ("libri", "audioset", "fma")}
    w(f"Real clips: {clips['libri']} LibriSpeech utterances (train-clean-100, the held-out last 2% of the sorted list, cut to 10 s), "
      f"{clips['audioset']} AudioSet and {clips['fma']} FMA 10 s segments with every frame forced non-speech, from the held-out last "
      "2% of the pool sources. Random inputs: Gaussian and uniform noise of 1,000 to 96,000 samples, batch 1 and 2.\n")
    w("## Heads against PyTorch\n")
    H = r["heads"]
    labels = {"reference_vs_torch_int8_taps": "float32-heads ONNX vs PyTorch head, both on the ONNX int8 taps (all 24 inputs)",
              "final_vs_torch_int8_taps": "shipped ONNX vs PyTorch head, both on the ONNX int8 taps (all 24 inputs)",
              "final_vs_reference": "shipped ONNX vs float32-heads ONNX (real clips)",
              "torch_int8_taps_vs_torch_float_taps": "PyTorch head on int8 taps vs on float-trunk taps (real clips): the trunk quantisation alone",
              "final_vs_torch_float_taps": "shipped ONNX vs PyTorch head on float-trunk taps, as trained (real clips)"}
    w("Differences are in probability. Decisions: vad p >= 0.5 per frame, ipa argmax per frame.\n")
    w("| comparison | vad max abs | vad agreement | ipa max abs | ipa agreement |")
    w("|---|---|---|---|---|")
    for k, lab in labels.items():
        v, i = H[k]["vad"], H[k]["ipa"]
        w(f"| {lab} | {v['max_abs_diff']:.2e} | {v['decision_agreement']:.5f} | {i['max_abs_diff']:.2e} | {i['decision_agreement']:.5f} |")
    w(f"\nFrames compared on real clips: {H['final_vs_torch_float_taps']['vad']['decisions']:,}.\n")
    va = r["vad_active_fraction_by_source"]
    w("VAD active fraction (p >= 0.5) on the real clips: " + "; ".join(f"{k} mean {v['mean']} (min {v['min']}, max {v['max']}, {v['clips']} clips)" for k, v in va.items()) + ".\n")
    m = r["misc"]
    w(f"Other checks: `features` equals the concatenation of its parts to {m['features_vs_concat_max_abs']}; the features-only file equals the "
      f"full file's `features` to {m['features_only_vs_full_max_abs']}; `logmel` vs PyTorch {m['logmel_vs_torch_max_abs']:.2e}; `mfcc` vs a "
      f"PyTorch DCT {m['mfcc_vs_torch_dct_max_abs']:.2e}; smallest `layers` value {m['layers_min']} (post-ReLU); frame count ONNX minus "
      f"float trunk {m['frames_onnx_minus_float_trunk']}.\n")
    w("## Latency\n")
    lk = [k for k in r if k.startswith("latency_")][0]
    w(f"One call on a 24,000-sample (1.5 s) waveform, as the plugin runs it every 1,280 samples (80 ms); onnxruntime "
      f"CPUExecutionProvider, one intra-op thread, sequential; 20 warm-up calls and 300 timed calls, two rounds. CPU: {r['cpu']}; "
      f"load average after the run {', '.join(f'{x:.1f}' for x in r['loadavg_after'])}.\n")
    w("| model | median ms (rounds) | p95 ms | share of the 80 ms block |")
    w("|---|---|---|---|")
    for k, v in r[lk].items():
        w(f"| {k} | {' / '.join(str(x['median_ms']) for x in v)} | {' / '.join(str(x['p95_ms']) for x in v)} | {v[0]['fraction_of_80ms_block']} |")
    w("\n## Source checkpoints\n")
    for h, v in cfg["heads"].items():
        w(f"- {h}: `{v['source_checkpoint']}` sha256 `{v['sha256']}`; hidden {v['hidden']}, side branch {v['side_branch_channels'] or 'none'}, "
          f"{v['params']:,} parameters, stored {v['dtype']}")
    w("\n## Commands\n")
    w("```")
    w(commands.read_text().strip())
    w("```")
    out_md.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
