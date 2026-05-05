#!/bin/bash
# ── Full-Horizon SFT (Ours) ────────────────────────────────────────────────
# Single-stage I2V training on complete episodes with uniform temporal coverage.
# Key difference from vanilla SFT: trim_mode=speedup covers the full manipulation
# causal chain instead of random snippets.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/main_exp/EVA-main_v0.5}"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-/home/zhangmohan/data/robotwin_sft_vanilla}"
WAN_CKPT="${WAN_CKPT:-/root/autodl-tmp/main_exp/Wan2.1-I2V-14B-480P}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

LR="${LR:-5e-6}"
BATCH="${BATCH:-1}"        # per-GPU: 1 (49 frames heavier than 17)
ACCUM="${ACCUM:-8}"        # effective BS = 1×8×8 = 64
MAX_STEPS="${MAX_STEPS:-50000}"
RESUME="${RESUME:-}"
USE_PRE_ENCODED="${USE_PRE_ENCODED:-false}"
N_FRAMES="${N_FRAMES:-49}"
LORA_RANK="${LORA_RANK:-32}"
METADATA_PATH="${METADATA_PATH:-metadata.csv}"

echo "=== Full-Horizon SFT (Ours) ==="
echo "Data: $DATA_ROOT | Wan: $WAN_CKPT | GPUs: $NPROC"
echo "LR=$LR BS=$BATCH×${NPROC}×$ACCUM steps=$MAX_STEPS frames=$N_FRAMES trim=speedup"

MASTER_PORT="${MASTER_PORT:-29500}"

torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" main.py \
    experiment=exp_video \
    experiment.strategy=ddp \
    dataset=robotwin_sft_full \
    algorithm=wan_i2v \
    +name=full_horizon_sft_$(date +%Y%m%d_%H%M) \
    \
    dataset.data_root="$DATA_ROOT" \
    dataset.metadata_path="$METADATA_PATH" \
    dataset.n_frames="$N_FRAMES" \
    dataset.load_video_latent="$USE_PRE_ENCODED" \
    dataset.load_prompt_embed="$USE_PRE_ENCODED" \
    \
    algorithm.model.ckpt_path="$WAN_CKPT" \
    algorithm.model.use_lora=true \
    algorithm.model.lora_rank="$LORA_RANK" \
    algorithm.model.lora_alpha="$LORA_RANK" \
    \
    algorithm.text_encoder.ckpt_path="$WAN_CKPT/models_t5_umt5-xxl-enc-bf16.pth" \
    algorithm.text_encoder.name="$WAN_CKPT/google/umt5-xxl" \
    algorithm.text_encoder.compile=false \
    algorithm.vae.ckpt_path="$WAN_CKPT/Wan2.1_VAE.pth" \
    algorithm.vae.compile=false \
    algorithm.clip.ckpt_path="$WAN_CKPT/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    algorithm.clip.compile=false \
    \
    experiment.training.lr="$LR" \
    experiment.training.batch_size="$BATCH" \
    experiment.training.max_steps="$MAX_STEPS" \
    experiment.training.optim.accumulate_grad_batches="$ACCUM" \
    experiment.training.checkpointing.every_n_train_steps=5000 \
    experiment.training.checkpointing.every_n_epochs=null \
    experiment.training.checkpointing.save_weights_only=false \
    experiment.training.checkpointing.enable_version_counter=true \
    experiment.tasks='[training]' \
    load="$RESUME" \
    \
    wandb.mode="${WANDB_MODE:-online}" \
    wandb.project="${WANDB_PROJECT:-wan-robot-sft}" \
    wandb.entity="${WANDB_ENTITY:-}" \
    +wandb.name="full_horizon_sft" \
    "$@"
