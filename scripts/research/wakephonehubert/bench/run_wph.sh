set -eu
BENCH=${1:-work/bench}
HEADS=${2:-work/runs/round4-final-ipa}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
L=$BENCH/logs
mkdir -p "$L"
cd "$HERE"
OPT="--bench-dir $BENCH --heads $HEADS"
$PY pr.py train wakephonehubert --workers 3 $OPT > $L/pr-wakephonehubert.log 2> $L/pr-wakephonehubert.err
$PY extract_pooled.py esc50 --feats wakephonehubert $OPT > $L/extract-esc50-wph.log 2> $L/extract-esc50-wph.err
$PY train_pooled.py esc50 wakephonehubert $OPT > $L/esc50-wakephonehubert.log 2>&1
$PY extract_pooled.py lid --feats wakephonehubert $OPT > $L/extract-lid-wph.log 2> $L/extract-lid-wph.err
$PY train_pooled.py lid wakephonehubert $OPT > $L/lid-wakephonehubert.log 2>&1
$PY extract_pooled.py sid --feats wakephonehubert $OPT > $L/extract-sid-wph.log 2> $L/extract-sid-wph.err
$PY train_pooled.py sid wakephonehubert $OPT > $L/sid-wakephonehubert.log 2>&1
$PY vad_eval.py --systems wakephonehubert $OPT > $L/vad-wph.log 2> $L/vad-wph.err
$PY speed.py --rows wakephonehubert $OPT > $L/speed-wph.log 2> $L/speed-wph.err
$PY report.py --bench-dir $BENCH > /dev/null
echo WPH_DONE
