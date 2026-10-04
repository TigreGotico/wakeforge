"""run_ctc.py [uniform wph_ctc teacher_ctc] [--work-dir ...] [--export-dir ...]

Approach A (WakePhoneHuBERT ipa + CTC forced alignment), the teacher through the same pipeline,
and the uniform floor, on test-clean. Writes results/<system>.jsonl."""
import argparse, time
import numpy as np, torch, onnxruntime as ort
import common
from common import *

ap = argparse.ArgumentParser()
common.add_args(ap)
ap.add_argument("systems", nargs="*")
args = ap.parse_args()
common.configure(args)
ROOT, WPH = common.ROOT, common.WPH
torch.set_num_threads(4)
systems = args.systems or ["uniform", "wph_ctc", "teacher_ctc"]
data = load_test()
ipa = IPA()
(ROOT / "results").mkdir(parents=True, exist_ok=True)


def base(u):
    w = u["words"]
    return dict(id=u["id"], words=u["text"], ref_start=[x["start"] for x in w],
                ref_end=[x["end"] for x in w], dur=len(u["audio"]) / 16000)


if "uniform" in systems:
    rows = []
    for u in data:
        r = base(u)
        a, b, n = r["ref_start"][0], r["ref_end"][-1], len(r["words"])
        edges = np.linspace(a, b, n + 1)
        r.update(pred_start=edges[:-1].round(4).tolist(), pred_end=edges[1:].round(4).tolist(),
                 note="equal word durations over the reference speech span (oracle span)")
        rows.append(r)
    write_jsonl(ROOT / "results" / "uniform.jsonl", rows)
    print("uniform", len(rows), flush=True)

if "wph_ctc" in systems or "teacher_ctc" in systems:
    toks = {u["id"]: ipa(u["text"]) for u in data}

if "wph_ctc" in systems:
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(WPH), so, providers=["CPUExecutionProvider"])
    rows, t0 = [], time.time()
    for u in data:
        r = base(u)
        p = sess.run(["ipa"], {"waveform": u["audio"][None]})[0][0]
        logp = torch.from_numpy(np.log(np.maximum(p, 1e-10)))
        try:
            sp = ctc_word_spans(logp, toks[u["id"]], DELAY_FRAMES)
            r.update(pred_start=[s for s, _ in sp], pred_end=[e for _, e in sp])
        except Exception as ex:
            r.update(pred_start=None, pred_end=None, error=repr(ex))
        r["ipa_ids"] = toks[u["id"]]
        rows.append(r)
    write_jsonl(ROOT / "results" / "wph_ctc.jsonl", rows)
    print("wph_ctc", len(rows), sum(r["pred_start"] is None for r in rows), "failed", f"{time.time()-t0:.0f}s", flush=True)

if "teacher_ctc" in systems:
    from transformers import Wav2Vec2ForCTC
    torch.cuda.set_per_process_memory_fraction(0.22)
    m = Wav2Vec2ForCTC.from_pretrained(TEACHER).cuda().eval()
    rows, t0 = [], time.time()
    for u in data:
        r = base(u)
        x = torch.from_numpy(u["audio"])
        x = ((x - x.mean()) / torch.sqrt(x.var() + 1e-7))[None].cuda()
        with torch.inference_mode():
            logp = m(x).logits[0].float().log_softmax(-1).cpu()
        try:
            sp = ctc_word_spans(logp, toks[u["id"]], 0)
            r.update(pred_start=[s for s, _ in sp], pred_end=[e for _, e in sp])
        except Exception as ex:
            r.update(pred_start=None, pred_end=None, error=repr(ex))
        rows.append(r)
    write_jsonl(ROOT / "results" / "teacher_ctc.jsonl", rows)
    print("teacher_ctc", len(rows), sum(r["pred_start"] is None for r in rows), "failed", f"{time.time()-t0:.0f}s",
          "peak GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
