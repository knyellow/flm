DATA_DIR="/app/data/flm_cache/tinystories"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=tinystories \
  data.cache_dir=$DATA_DIR \
  wandb.project=tinystories_flm \
  wandb.name=tinystories_flm \
  model=small \
  algo=flm \
  model.length=512 \
  sampling.num_sample_batches=1 \
  sampling.solver=euler \
  sampling.steps=[128] \
  trainer.max_steps=1024 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=150 \
  algo.double_temb=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=150