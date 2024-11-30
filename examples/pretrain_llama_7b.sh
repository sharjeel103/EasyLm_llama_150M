#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='aa9f4011859de3cc2776a7a9d48b8e3d81466b5c'

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python -m EasyLM.models.llama.llama_train \
    --load_checkpoint='' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=6e-4 \
    --optimizer.adamw_optimizer.end_lr=6e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=500 \
    --optimizer.adamw_optimizer.lr_decay_steps=14500 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='smart_toy' \
    --logger.project="llama_150m" \
    --logger.output_dir="gs://sharjeel/smart_toy/" \
    --logger.wandb_dir="$HOME/first/llama_150m" \
|& tee $HOME/output.txt

