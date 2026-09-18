
WW="hey_mycroft"

ARCH="gru" # ffn / cnn

# Where the dataset, the featurizer and the augmentation folders live.
# Override from the environment; the defaults are relative to the repo root.
DATA_ROOT="${WW_DATA_ROOT:-data/ww}"
FEATURIZER="${WW_FEATURIZER:-scripts/distillhubert_int8.onnx}"
AUG_ROOT="${WW_AUG_ROOT:-data/augmentation}"


python ww_trainer/trainer.py \
  --wake-word $WW \
  --metadata $DATA_ROOT/$WW/train/metadata.csv \
  --test-metadata $DATA_ROOT/$WW/test/metadata.csv \
  --featurizer $FEATURIZER \
  --export-onnx \
  --feature-dim 768 \
  --epochs 50 \
  --batch-size 1000 \
  --mine-sample 0.2 \
  --arch $ARCH \
  --loss-type rppl \
  --loss-weight 1.0 \
  --aug-prob 0.5 \
  --mining-type semihard \
  --device cpu \
  --pca-every 1 \
  --tsne-every 2 \
  --umap-every 5 \
  --base-hard 1.0 \
  --max-hard 10.0 \
  --base-easy 5.0 \
  --min-easy 1.0 \
  --total-ratio 10.0 \
  --mlflow-uri http://localhost:5000 \
  --bg-noise-folder $AUG_ROOT/building_106_kitchen_3secs \
  --mic-noise-folder $AUG_ROOT/bk \
  --rir-folder $AUG_ROOT/MIT_environmental_impulse-responses \
  --music-folder $AUG_ROOT/FMA_3secs