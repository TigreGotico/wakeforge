set -eu
WORK=${1:-work}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
POOLPY=$HERE/../pool
POOL=$WORK/pool
RUNS=$WORK/runs
LOGS=$WORK/logs
LIBRI=${LIBRI:-$WORK/data/LibriSpeech/train-clean-100}
NOISE=${NOISE:-$WORK/data/noise}
RIR=${RIR:-$WORK/data/rir}
ONTOLOGY=${ONTOLOGY:-$WORK/data/audioset/ontology.json}
EFFICIENTAT=${EFFICIENTAT:-$WORK/EfficientAT}
EXCLUDE_TRACKS=${EXCLUDE_TRACKS:-}
AUDIOSET_EVAL_DIR=${AUDIOSET_EVAL_DIR:-}
mkdir -p "$RUNS" "$LOGS"

TEACH="--ontology $ONTOLOGY --efficientat $EFFICIENTAT"
TRAIN="--libri $LIBRI --rir $RIR --side 128"
[ -n "$EXCLUDE_TRACKS" ] && TRAIN="$TRAIN --exclude-tracks $EXCLUDE_TRACKS"
[ -n "$AUDIOSET_EVAL_DIR" ] && TRAIN="$TRAIN --audioset-eval-dir $AUDIOSET_EVAL_DIR"

$PY $HERE/train_vad.py vad-data $RUNS/vad-data --speech $LIBRI --noise $NOISE > $LOGS/vad-data.log 2>&1
$PY $HERE/train_vad.py vad $RUNS/vad-data $RUNS/vad > $LOGS/vad.log 2>&1

$PY $POOLPY/snapshot_pool.py $POOL mswc mswc_a --type mswc --count 260964
$PY $POOLPY/pool_teachers.py $POOL --source fma $TEACH > $LOGS/pt-fma.log 2>&1
$PY $POOLPY/pool_teachers.py $POOL --source mswc_a --type mswc $TEACH > $LOGS/pt-mswc_a.log 2>&1
$PY $HERE/train_pool.py $POOL $RUNS/round1 --sources fma,mswc_a --epochs 6 --steps-per-epoch 3000 --init-vad $RUNS/vad/vad_head.pt $TRAIN > $LOGS/round1.log 2>&1
echo ROUND1_DONE

$PY $POOLPY/snapshot_pool.py $POOL mswc mswc_b --type mswc --start 260964
$PY $POOLPY/snapshot_pool.py $POOL audioset audioset_a --type audioset --count 48000
$PY $POOLPY/pool_teachers.py $POOL --source mswc_b --type mswc $TEACH > $LOGS/pt-mswc_b.log 2>&1
$PY $POOLPY/pool_teachers.py $POOL --source audioset_a --type audioset $TEACH > $LOGS/pt-audioset_a.log 2>&1
$PY $HERE/train_pool.py $POOL $RUNS/round2 --sources fma,audioset_a,mswc_a,mswc_b --epochs 6 --steps-per-epoch 3000 --resume $RUNS/round1 $TRAIN > $LOGS/round2.log 2>&1
echo ROUND2_DONE

$PY $POOLPY/snapshot_pool.py $POOL audioset audioset_b --type audioset --start 48000
$PY $POOLPY/pool_teachers.py $POOL --source audioset_b --type audioset $TEACH > $LOGS/pt-audioset_b.log 2>&1
$PY $POOLPY/pool_teachers.py $POOL --source mswcx --type mswc --stages text > $LOGS/pt-mswcx.log 2>&1
$PY $HERE/train_pool.py $POOL $RUNS/round3 --sources fma,audioset_a,audioset_b,mswc_a,mswc_b --epochs 4 --steps-per-epoch 3000 --resume $RUNS/round2 $TRAIN > $LOGS/round3.log 2>&1
echo ROUND3_DONE

$PY $HERE/train_pool.py $POOL $RUNS/round4-final-ipa --sources fma,audioset_a,audioset_b,mswc_a,mswc_b --resume $RUNS/round3 --final-ipa-sources mswcx --epochs 2 --steps-per-epoch 5000 $TRAIN > $LOGS/round4.log 2>&1
echo ALL_DONE
