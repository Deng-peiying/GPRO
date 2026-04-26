#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${DEBUG_SHELL:-0}" == "1" ]]; then
  set -x
fi

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2' ERR

# Reference-model GRPO smoke run.
# Default placement:
#   policy Wan     -> cuda:0
#   reference Wan  -> cuda:1
#   DA3 / AnyPos   -> cuda:0
#
# Usage:
#   conda activate grpo_wan
#   source scripts/run_wan_grpo_ref_smoke.sh
#
# Optional overrides:
#   GROUP_SIZE=2 STEPS=20 LR=5e-6 source scripts/run_wan_grpo_ref_smoke.sh

export PYTHONPATH="/root/autodl-tmp/repos/EVA-main:/root/autodl-tmp/repos/Depth-Anything-3-main/src:/root/autodl-tmp/repos/AnyPos-main:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/repos/EVA-main}"
CONFIG_PATH="${CONFIG_PATH:-/root/autodl-tmp/repos/EVA-main/configurations/algorithm/wan_i2v.yaml}"
CONDITION_BANK="${CONDITION_BANK:-/root/autodl-tmp/wan_grpo_condition_bank.pt}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/grpo_smoke_ref}"

ANYPOS_CKPT="${ANYPOS_CKPT:-/root/autodl-tmp/models/AnyPos/anypos_model.pt}"
DINO_DIR="${DINO_DIR:-/root/autodl-tmp/models/dinov2-with-registers-base}"
DA3_DIR="${DA3_DIR:-/root/autodl-tmp/models/DA3-LARGE-1.1}"
DA3_REPO_ROOT="${DA3_REPO_ROOT:-/root/autodl-tmp/repos/Depth-Anything-3-main}"

WAN_DEVICE="${WAN_DEVICE:-cuda:0}"
REF_DEVICE="${REF_DEVICE:-cuda:1}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda:0}"
IDM_DEVICE="${IDM_DEVICE:-cuda:0}"

STEPS="${STEPS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
GROUP_SIZE="${GROUP_SIZE:-2}"
HORIZON_STEPS="${HORIZON_STEPS:-2}"
HIST_LEN="${HIST_LEN:-1}"
FLOW_SAMPLING_NOISE_STD="${FLOW_SAMPLING_NOISE_STD:-0.08}"

LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP_EPS="${CLIP_EPS:-0.1}"
BETA_KL="${BETA_KL:-1e-3}"
SURROGATE_SIGMA="${SURROGATE_SIGMA:-0.5}"
LOG_RATIO_CLIP="${LOG_RATIO_CLIP:-2.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
REF_UPDATE_INTERVAL="${REF_UPDATE_INTERVAL:-0}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
GRADIENT_CHECKPOINTING_RATE="${GRADIENT_CHECKPOINTING_RATE:-1.0}"

OVERRIDE_HEIGHT="${OVERRIDE_HEIGHT:-192}"
OVERRIDE_WIDTH="${OVERRIDE_WIDTH:-256}"
OVERRIDE_N_FRAMES="${OVERRIDE_N_FRAMES:-5}"
OVERRIDE_SAMPLE_STEPS="${OVERRIDE_SAMPLE_STEPS:-4}"

HARD_VETO_PENALTY="${HARD_VETO_PENALTY:-50.0}"
MAX_CONTROL_DELTA="${MAX_CONTROL_DELTA:-0.25}"
FEASIBILITY_WEIGHT="${FEASIBILITY_WEIGHT:-1.0}"
ACTION_RECOVERY_WEIGHT="${ACTION_RECOVERY_WEIGHT:-1.0}"
IDM_STABILITY_WEIGHT="${IDM_STABILITY_WEIGHT:-0.1}"
DOF_WEIGHTS="${DOF_WEIGHTS:-1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}"

cd "${REPO_ROOT}"

python -m algorithms.wan.run_state_unrolled_grpo \
  --config "${CONFIG_PATH}" \
  --condition-bank "${CONDITION_BANK}" \
  --save-dir "${SAVE_DIR}" \
  --steps "${STEPS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --group-size "${GROUP_SIZE}" \
  --horizon-steps "${HORIZON_STEPS}" \
  --hist-len "${HIST_LEN}" \
  --flow-sampling-noise-std "${FLOW_SAMPLING_NOISE_STD}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --clip-eps "${CLIP_EPS}" \
  --beta-kl "${BETA_KL}" \
  --surrogate-sigma "${SURROGATE_SIGMA}" \
  --log-ratio-clip "${LOG_RATIO_CLIP}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --ref-update-interval "${REF_UPDATE_INTERVAL}" \
  --use-reference-model \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --gradient-checkpointing-rate "${GRADIENT_CHECKPOINTING_RATE}" \
  --idm-checkpoint "${ANYPOS_CKPT}" \
  --idm-backend anypos \
  --idm-model-name direction_aware_with_split \
  --idm-dinov2-name "${DINO_DIR}" \
  --idm-left-arm-dim 6 \
  --idm-right-arm-dim 6 \
  --idm-model-output-dim 16 \
  --depth-backend da3 \
  --da3-model-dir "${DA3_DIR}" \
  --da3-repo-root "${DA3_REPO_ROOT}" \
  --wan-device "${WAN_DEVICE}" \
  --ref-device "${REF_DEVICE}" \
  --depth-device "${DEPTH_DEVICE}" \
  --idm-device "${IDM_DEVICE}" \
  --override-height "${OVERRIDE_HEIGHT}" \
  --override-width "${OVERRIDE_WIDTH}" \
  --override-n-frames "${OVERRIDE_N_FRAMES}" \
  --override-sample-steps "${OVERRIDE_SAMPLE_STEPS}" \
  --disable-model-compile \
  --disable-clip-compile \
  --disable-vae-compile \
  --disable-text-encoder-compile \
  --hard-veto-penalty "${HARD_VETO_PENALTY}" \
  --max-control-delta "${MAX_CONTROL_DELTA}" \
  --feasibility-weight "${FEASIBILITY_WEIGHT}" \
  --action-recovery-weight "${ACTION_RECOVERY_WEIGHT}" \
  --idm-stability-weight "${IDM_STABILITY_WEIGHT}" \
  --dof-weights "${DOF_WEIGHTS}"
