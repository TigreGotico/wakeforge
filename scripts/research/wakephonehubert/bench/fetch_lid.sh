set -u
D=${1:-work/bench/data}/voxlingua107/train
mkdir -p $D
for lang in sv nl hy fr fa ar da lv fi et; do
  for shard in 000000 000001; do
    free=$(df --output=avail -BG $D | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 10 ]; then echo "STOP: disk ${free}G"; exit 3; fi
    f=$D/$lang-$shard.tar
    if [ ! -f $f ]; then
      curl -sfL -o $f.part https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds/resolve/main/train/$lang/$shard.tar && mv $f.part $f
      echo "$(date -u +%FT%TZ) $lang $shard $(stat -c %s $f 2>/dev/null) free ${free}G"
      sleep 6
    fi
  done
done
echo FETCH_LID_DONE
