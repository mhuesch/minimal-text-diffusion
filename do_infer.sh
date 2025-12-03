source env/bin/activate

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# coverage run --append --source=src --omit="*/__init__.py" src/train_infer/text_sample.py \
python src/train_infer/text_sample.py \
  --model_name_or_path ./checkpoints/model030000.pt \
  --num_samples 8 \
  --top_p 0.9 \
  --diffusion_steps 1000