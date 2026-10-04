set -eu
WORK=${1:-work}
TRUNK=${2:-$WORK/wakehubert_int8.onnx}
HEADS=${3:-$WORK/runs/round4-final-ipa}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
LIBRI=${LIBRI:-$WORK/data/LibriSpeech/train-clean-100}
OUT=$WORK/export/final
CLIPS=$WORK/export/clips
mkdir -p "$WORK/export"
cd "$HERE"

$PY prep_clips.py $WORK/pool $LIBRI $CLIPS
CUDA_VISIBLE_DEVICES= $PY build_wakephonehubert.py --int8 $TRUNK --heads $HEADS --out $OUT --clips $CLIPS > $WORK/export/build.log 2>&1
CUDA_VISIBLE_DEVICES= $PY validate_wakephonehubert.py --build $OUT --int8 $TRUNK --heads $HEADS --clips $CLIPS/val > $WORK/export/validate.log 2>&1
printf '%s\n' "$PY prep_clips.py $WORK/pool $LIBRI $CLIPS" "$PY build_wakephonehubert.py --int8 $TRUNK --heads $HEADS --out $OUT --clips $CLIPS" "$PY validate_wakephonehubert.py --build $OUT --int8 $TRUNK --heads $HEADS --clips $CLIPS/val" > $WORK/export/commands.txt
$PY write_validation_md.py $OUT $OUT/VALIDATION.md $CLIPS/manifest.json $WORK/export/commands.txt
echo EXPORT_DONE
