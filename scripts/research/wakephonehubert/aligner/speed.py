"""speed.py [--work-dir ...] [--export-dir ...]

CPU time per second of audio, one thread, for approach A, approach B and the teacher.
Process CPU time (time.process_time) over a fixed sample of test-clean clips; phonemisation excluded."""
import argparse, json, os, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch, onnxruntime as ort
import common
from common import *
from head import viterbi_align

ap = argparse.ArgumentParser()
common.add_args(ap)
common.configure(ap.parse_args())
ROOT, WPH = common.ROOT, common.WPH

torch.set_num_threads(1); torch.set_num_interop_threads(1)
data = load_test()[::26]
ipa = IPA()
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
wph = ort.InferenceSession(str(WPH), so, providers=["CPUExecutionProvider"])
hd = ort.InferenceSession(str(ROOT / "head" / "aligner_head.onnx"), so, providers=["CPUExecutionProvider"])
toks = [ipa(u["text"]) for u in data]
seqs = []
for u in data:
    g = word_phones(u["words"], u["phonemes"])
    seqs.append([[ARPA_ID[strip_stress(p["phoneme"])] for p in w] for w in g])
audio_s = sum(len(u["audio"]) for u in data) / 16000
res = dict(clips=len(data), audio_seconds=audio_s, load_before=os.getloadavg())


def timed(fn, items):
    fn(*items[0])
    t = time.process_time()
    for it in items:
        fn(*it)
    return (time.process_time() - t) / audio_s


def a(u, tk):
    p = wph.run(["ipa"], {"waveform": u["audio"][None]})[0][0]
    ctc_word_spans(torch.from_numpy(np.log(np.maximum(p, 1e-10))), tk, DELAY_FRAMES)


def b(u, sq):
    L, F = wph.run(["layers", "features"], {"waveform": u["audio"][None]})
    lp = hd.run(["logp"], {"layers": L, "features": F})[0][0]
    viterbi_align(lp, sq)


res["approach_A_cpu_s_per_audio_s"] = timed(a, list(zip(data, toks)))
res["approach_B_cpu_s_per_audio_s"] = timed(b, list(zip(data, seqs)))

from transformers import Wav2Vec2ForCTC
m = Wav2Vec2ForCTC.from_pretrained(TEACHER).eval()


def te(u, tk):
    x = torch.from_numpy(u["audio"]); x = ((x - x.mean()) / torch.sqrt(x.var() + 1e-7))[None]
    with torch.inference_mode():
        lp = m(x).logits[0].log_softmax(-1)
    ctc_word_spans(lp, tk, 0)


res["teacher_cpu_s_per_audio_s"] = timed(te, list(zip(data, toks)))
res["load_after"] = os.getloadavg()
json.dump(res, open(ROOT / "results" / "speed.json", "w"), indent=1)
print(json.dumps(res, indent=1))
