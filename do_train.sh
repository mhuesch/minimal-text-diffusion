source env/bin/activate

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# coverage run --append --source=src --omit="*/__init__.py" src/train_infer/train.py \
python src/train_infer/train.py \
  --train_txt_path data/e2e_train.txt \
  --val_txt_path   data/e2e_val.txt \
  --dataset e2e \
  --sequence_len 50 \
  --batch_size 32 \
  --modality text \
  --training_mode diffusion-lm \
  --checkpoint_path ./checkpoints

#  --train_txt_path data/simple-train.txt \
#  --val_txt_path data/simple-test.txt \
#  --dataset simple \