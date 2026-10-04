set -eu
WORK=${1:-work}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
POOL=$WORK/pool
mkdir -p "$POOL" "$WORK/logs"
cd "$HERE"

$PY build_pool.py $POOL fma --hours 400 --fma-metadata $WORK/data/fma_metadata.zip > $WORK/logs/pool-fma.log 2>&1
$PY build_pool.py $POOL audioset --shards 40 > $WORK/logs/pool-audioset.log 2>&1

$PY build_pool.py $POOL mswc --per-lang 60000 --langs ar,as,br,ca,cnh,cs,cv,cy > $WORK/logs/pool-mswc-first-languages.log 2>&1
$PY build_pool.py $POOL mswc --per-lang 20000 > $WORK/logs/pool-mswc.log 2>&1

$PY build_pool.py $POOL mswc --name mswcx --per-lang 25000 --shards-per-lang 3 --shard-offset 0.5 > $WORK/logs/pool-mswcx.log 2>&1
echo POOL_DONE
