source env/bin/activate

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# coverage run --append --source=src --omit="*/__init__.py" src/train_infer/text_sample.py \
python src/train_infer/text_sample.py \
  --model_name_or_path ./checkpoints/ema_0.9999_030000.pt \
  --num_samples 8 \
  --diffusion_steps 4000 \
  --sequence_len 50 \
  --use_bert_tokenizer True