
WW="hey_mycroft"

ARCH="gru" # ffn / cnn


python ww_trainer/trainer.py \
  --wake-word $WW \
  --metadata /run/media/miro/endeavouros/ww/$WW/train/metadata.csv \
  --test-metadata /run/media/miro/endeavouros/ww/$WW/test/metadata.csv \
  --featurizer /home/miro/PycharmProjects/ww-trainer/scripts/distillhubert_int8.onnx \
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
  --mlflow-uri http://192.168.1.200:5000 \
  --bg-noise-folder /run/media/miro/701e86b2-c4d8-47d2-969a-c64510db39e9/building_106_kitchen_3secs \
  --mic-noise-folder /run/media/miro/701e86b2-c4d8-47d2-969a-c64510db39e9/bk \
  --rir-folder /run/media/miro/701e86b2-c4d8-47d2-969a-c64510db39e9/MIT_environmental_impulse-responses \
  --music-folder /run/media/miro/701e86b2-c4d8-47d2-969a-c64510db39e9/FMA_3secs