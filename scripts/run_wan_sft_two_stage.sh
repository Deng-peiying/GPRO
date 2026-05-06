#!/bin/bash
# ── C: Two-stage SFT ────────────────────────────────────────────────────
# Stage 2: resumes from entry-focused checkpoint, trains on full-horizon videos.
# Run AFTER run_wan_sft_entry.sh completes.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/main_exp/EVA-main_v0.5}"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-/home/zhangmohan/data/robotwin_sft_vanilla}"
WAN_CKPT="${WAN_CKPT:-/root/autodl-tmp/main_exp/Wan2.1-I2V-14B-480P}"

# REQUIRED: path to entry-focused SFT checkpoint
ENTRY_CKPT="${ENTRY_CKPT:?Must set ENTRY_CKPT to entry-focused checkpoint path}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

LR="${LR:-3e-6}"          # lower LR for full-horizon
BATCH="${BATCH:-1}"        # smaller per-GPU batch (49 frames)
ACCUM="${ACCUM:-8}"        # compensate with accumulation
MAX_STEPS="${MAX_STEPS:-50000}"
RESUME="${RESUME:-}"  # set RESUME=/path/to/checkpoint.ckpt to resume
USE_PRE_ENCODED="${USE_PRE_ENCODED:-false}"  # set to true after running pre_encode_robotwin_data.py
N_FRAMES="${N_FRAMES:-49}" # full horizon
LORA_RANK="${LORA_RANK:-32}"

echo "=== C: Two-stage SFT (Stage 2: full-horizon) ==="
echo "Entry ckpt: $ENTRY_CKPT"
echo "Data: $DATA_ROOT | Wan: $WAN_CKPT | GPUs: $NPROC"
echo "LR=$LR BS=$BATCH×${NPROC}×$ACCUM steps=$MAX_STEPS frames=$N_FRAMES"

MASTER_PORT="${MASTER_PORT:-29500}"

torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" main.py \
    experiment=exp_video \
    experiment.strategy=ddp \
    dataset=robotwin_sft_full \
    algorithm=wan_i2v \
    +name=two_stage_sft_$(date +%Y%m%d_%H%M) \
    \
    dataset.data_root="$DATA_ROOT" \
    dataset.n_frames="$N_FRAMES" \
    dataset.load_video_latent="$USE_PRE_ENCODED" \
    dataset.load_prompt_embed="$USE_PRE_ENCODED" \
    \
    algorithm.model.ckpt_path="$WAN_CKPT" \
    algorithm.model.tuned_ckpt_path="$ENTRY_CKPT" \
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
wandb.mode="${WANDB_MODE:-disabled}" \
    wandb.project="${WANDB_PROJECT:-wan-robot-sft}" \
    +wandb.name="C_two_stage_sft"
