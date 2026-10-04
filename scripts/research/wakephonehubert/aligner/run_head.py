"""run_head.py [--work-dir ...] [--export-dir ...]

Approach B on test-clean: WakePhoneHuBERT -> aligner head -> constrained Viterbi.
Two phone-sequence sources: the MFA reference phones (head_ref) and the first CMUdict
pronunciation of each word (head_cmudict; a word CMUdict lacks becomes the single class spn).
Also exports the head to ONNX and saves it as safetensors."""
import argparse
import numpy as np, torch, onnxruntime as ort
from safetensors.torch import save_file
import common
from common import *
from head import AlignHead, viterbi_align

ap = argparse.ArgumentParser()
common.add_args(ap)
common.configure(ap.parse_args())
ROOT, WPH = common.ROOT, common.WPH

model = AlignHead().eval()
model.load_state_dict(torch.load(ROOT / "head" / "aligner_head.pt", map_location="cpu"))
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, ROOT / "head" / "aligner_head.safetensors")

L = torch.rand(1, 120, 9, 256) * 3; F = torch.rand(1, 120, 521)
onnx_path = ROOT / "head" / "aligner_head.onnx"
torch.onnx.export(model, (L, F), str(onnx_path), input_names=["layers", "features"], output_names=["logp"],
                  dynamic_axes={"layers": {0: "batch", 1: "frames"}, "features": {0: "batch", 1: "frames"},
                                "logp": {0: "batch", 1: "frames"}}, opset_version=17, dynamo=False)

cmu = {}
for line in open(ROOT / "data" / "cmudict.dict"):
    w, *ph = line.split("#")[0].split()
    if "(" not in w:
        cmu.setdefault(w, [ARPA_ID[strip_stress(p)] for p in ph])

so = ort.SessionOptions(); so.intra_op_num_threads = 4
wph = ort.InferenceSession(str(WPH), so, providers=["CPUExecutionProvider"])
hd = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
data = load_test()
rows_ref, rows_cmu, oov, maxdiff = [], [], 0, 0.0
for n, u in enumerate(data):
    Lw, Fw = wph.run(["layers", "features"], {"waveform": u["audio"][None]})
    lp = hd.run(["logp"], {"layers": Lw, "features": Fw})[0][0]
    if n < 20:
        with torch.no_grad():
            ref = model(torch.from_numpy(Lw), torch.from_numpy(Fw))[0].numpy()
        maxdiff = max(maxdiff, float(np.abs(np.exp(ref) - np.exp(lp)).max()))
    groups = word_phones(u["words"], u["phonemes"])
    seq_ref = [[ARPA_ID[strip_stress(p["phoneme"])] for p in g] for g in groups]
    seq_cmu = []
    for w, s in zip(u["words"], seq_ref):
        c = cmu.get(w["word"].lower())
        oov += c is None
        seq_cmu.append(c if c is not None else [ARPA_ID["spn"]])
    base = dict(id=u["id"], words=[w["word"] for w in u["words"]], ref_start=[w["start"] for w in u["words"]],
                ref_end=[w["end"] for w in u["words"]], dur=len(u["audio"]) / 16000)
    for seq, rows, extra in ((seq_ref, rows_ref, True), (seq_cmu, rows_cmu, False)):
        al = viterbi_align(lp, seq)
        r = dict(base, pred_start=[w[0][1] for w in al], pred_end=[w[-1][2] for w in al])
        r["pred_phones"] = [[ARPA[c], round(s, 3), round(e, 3)] for w in al for c, s, e in w]
        if extra:
            r["ref_phones"] = [[strip_stress(p["phoneme"]), p["start"], p["end"]] for g in groups for p in g]
        rows.append(r)
write_jsonl(ROOT / "results" / "head_ref.jsonl", rows_ref)
write_jsonl(ROOT / "results" / "head_cmudict.jsonl", rows_cmu)
print("utts", len(data), "cmudict OOV words", oov, "onnx vs torch max prob diff (20 utts)", maxdiff)
