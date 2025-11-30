source env/bin/activate

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python src/train_infer/train.py \
  --train_txt_path data/simple-train.txt \
  --val_txt_path data/simple-test.txt \
  --dataset simple \
  --sequence_len 50 \
  --batch_size 32 \
  --modality text \
  --training_mode diffusion-lm \
  --checkpoint_path ./checkpoints