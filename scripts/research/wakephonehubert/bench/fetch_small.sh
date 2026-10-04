set -u
D=${1:-work/bench/data}
mkdir -p $D/esc50 $D/voxceleb1 $D/voxlingua107 $D/libriparty
guard() {
  free=$(df --output=avail -BG $D | tail -1 | tr -dc 0-9)
  echo "$(date -u +%FT%TZ) disk free ${free}G"
  if [ "$free" -lt 20 ]; then echo "STOP: disk under 20G"; exit 3; fi
}
guard
[ -f $D/voxceleb1/iden_split.txt ] || curl -sfL -o $D/voxceleb1/iden_split.txt https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt
sleep 5
[ -f $D/voxceleb1/vox1_meta.csv ] || curl -sfL -o $D/voxceleb1/vox1_meta.csv https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_meta.csv
sleep 5
if [ ! -f $D/esc50/ESC-50-master.zip ]; then
  curl -sfL -o $D/esc50/ESC-50-master.zip.part https://github.com/karoldvl/ESC-50/archive/master.zip && mv $D/esc50/ESC-50-master.zip.part $D/esc50/ESC-50-master.zip
  (cd $D/esc50 && unzip -q ESC-50-master.zip)
fi
guard
sleep 5
if [ ! -f $D/voxlingua107/dev.tar ]; then
  curl -sfL -o $D/voxlingua107/dev.tar.part https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds/resolve/main/dev/dev.tar && mv $D/voxlingua107/dev.tar.part $D/voxlingua107/dev.tar
fi
guard
sleep 5
if [ ! -f $D/libriparty/.done ]; then
  curl -sfL "https://www.dropbox.com/s/8zcn6zx4fnxvfyt/LibriParty.tar.gz?dl=1" | tar -xz -C $D/libriparty --wildcards '*/eval/*' '*/dev/*' && touch $D/libriparty/.done
fi
guard
echo FETCH_SMALL_DONE
