set -eu
BENCH=${1:-work/bench}
EAT=${2:-work/EfficientAT}
LIBRI=${3:-work/data/LibriSpeech}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
L=$BENCH/logs
mkdir -p "$L"
cd "$HERE"
OPT="--bench-dir $BENCH --efficientat $EAT"
FEATS=logmel,wakehubert,hubert-base,wav2vec2-espeak,efficientat-mn10
$PY extract_pooled.py esc50 --feats $FEATS --threads 2 $OPT >> $L/extract-esc50.log 2>> $L/extract-esc50.err
for f in logmel wakehubert hubert-base wav2vec2-espeak efficientat-mn10; do
  $PY train_pooled.py esc50 $f $OPT > $L/esc50-$f.log 2>&1
done
$PY extract_pooled.py lid --feats $FEATS --threads 2 $OPT >> $L/extract-lid.log 2>> $L/extract-lid.err
for f in logmel wakehubert hubert-base wav2vec2-espeak efficientat-mn10; do
  $PY train_pooled.py lid $f $OPT > $L/lid-$f.log 2>&1
done
$PY extract_pooled.py sid --feats $FEATS --threads 2 $OPT >> $L/extract-sid.log 2>> $L/extract-sid.err
for f in logmel wakehubert efficientat-mn10 hubert-base wav2vec2-espeak; do
  $PY train_pooled.py sid $f $OPT > $L/sid-$f.log 2>&1
done
$PY pr.py prep --librispeech $LIBRI $OPT
$PY pr.py direct $OPT > $L/pr-wav2vec2-espeak-direct.log 2> $L/pr-wav2vec2-espeak-direct.err
for f in logmel wakehubert hubert-base wav2vec2-espeak; do
  $PY pr.py train $f --workers 3 $OPT > $L/pr-$f.log 2> $L/pr-$f.err
done
$PY vad_eval.py --systems silero,webrtc $OPT > $L/vad.log 2> $L/vad.err
$PY speed.py --rows logmel,wakehubert,wakehubert-onnx-int8,hubert-base,wav2vec2-espeak,efficientat-mn10,silero,webrtc $OPT > $L/speed.log 2> $L/speed.err
$PY report.py --bench-dir $BENCH > /dev/null
echo BASELINES_DONE
