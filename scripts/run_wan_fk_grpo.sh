#!/bin/bash
# ── D: Two-stage SFT + FK GRPO ──────────────────────────────────────────
# FK-grounded GRPO post-training from SFT checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/main_exp/EVA-main_v0.5}"
cd "$PROJECT_ROOT"

# ── Required paths ──────────────────────────────────────────────────────
CONDITION_BANK="${CONDITION_BANK:-/root/autodl-tmp/main_exp/wan_grpo_condition_bank_triple.pt}"
TWO_STAGE_CKPT="${TWO_STAGE_CKPT:-/root/autodl-tmp/main_exp/Ours_sft/checkpoints/robot-wan-epoch=55-step=3000.ckpt}"
WAN_CONFIG="${WAN_CONFIG:-$PROJECT_ROOT/configurations/algorithm/wan_i2v.yaml}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/outputs/fk_grpo}"
FK_URDF="${FK_URDF:-/root/autodl-tmp/main_exp/RoboTwin/assets/embodiments/franka-panda/panda.urdf}"
IDM_CHECKPOINT="${IDM_CHECKPOINT:-/root/autodl-tmp/main_exp/commit-freeze_test/output/full/152000.pt}"
VIDAR_REPO_ROOT="${VIDAR_REPO_ROOT:-/root/autodl-tmp/main_exp/commit-freeze_test}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-/root/autodl-tmp/main_exp/Depth-Anything-3-main/model/DA3-LARGE-1.1}"
DA3_REPO_ROOT="${DA3_REPO_ROOT:-/root/autodl-tmp/main_exp/Depth-Anything-3-main}"
export PYTHONPATH="$PROJECT_ROOT:$DA3_REPO_ROOT/src:$VIDAR_REPO_ROOT:${PYTHONPATH:-}"

# ── GPU ─────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-32}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
WAN_DEVICE="${WAN_DEVICE:-cuda:0}"
REF_DEVICE="${REF_DEVICE:-cuda:1}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda:3}"
IDM_DEVICE="${IDM_DEVICE:-cuda:3}"

# ── GRPO hyperparams ────────────────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-4}"
HORIZON="${HORIZON:-4}"
HIST_LEN="${HIST_LEN:-1}"
GRPO_STEPS="${GRPO_STEPS:-800}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
LR="${LR:-1e-6}"
NUM_INNER_EPOCHS="${NUM_INNER_EPOCHS:-${INNER_EPOCHS:-1}}"
CLIP_EPS="${CLIP_EPS:-0.2}"
BETA_KL="${BETA_KL:-0.01}"
SURROGATE_SIGMA="${SURROGATE_SIGMA:-1.0}"
LOG_RATIO_CLIP="${LOG_RATIO_CLIP:-5.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
REF_UPDATE_INTERVAL="${REF_UPDATE_INTERVAL:-0}"
DISCOUNT_GAMMA="${DISCOUNT_GAMMA:-1.0}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

# Video shape. The default condition bank is the triple-camera layout:
# head 640x480 + left/right 320x240 below -> 640x720, full horizon 49 frames.
N_FRAMES="${N_FRAMES:-49}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-640}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"

# FK reward weights
FK_WEIGHT="${FK_WEIGHT:-0.5}"
FK_WS_WEIGHT="${FK_WS_WEIGHT:-1.0}"
FK_SING_WEIGHT="${FK_SING_WEIGHT:-0.5}"
FK_EEVEL_WEIGHT="${FK_EEVEL_WEIGHT:-0.3}"
FK_EEACC_WEIGHT="${FK_EEACC_WEIGHT:-0.2}"
FK_CHAIN_WEIGHT="${FK_CHAIN_WEIGHT:-1.0}"
FK_DUAL_WEIGHT="${FK_DUAL_WEIGHT:-0.3}"

echo "=== D: FK GRPO ==="
echo "SFT ckpt: $TWO_STAGE_CKPT"
echo "Condition bank: $CONDITION_BANK"
echo "FK URDF: $FK_URDF"
echo "IDM repo: $VIDAR_REPO_ROOT"
echo "Save dir: $SAVE_DIR"
echo "Devices: WAN=$WAN_DEVICE REF=$REF_DEVICE DEPTH=$DEPTH_DEVICE IDM=$IDM_DEVICE"
echo "Video shape: ${N_FRAMES} frames @ ${WIDTH}x${HEIGHT}, sample_steps=$SAMPLE_STEPS"
echo "LoRA: rank=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"

python -m algorithms.wan.run_state_unrolled_grpo \
    --config "$WAN_CONFIG" \
    --condition-bank "$CONDITION_BANK" \
    --save-dir "$SAVE_DIR" \
    --tuned-ckpt "$TWO_STAGE_CKPT" \
    --idm-backend vidar \
    --idm-checkpoint "$IDM_CHECKPOINT" \
    --depth-backend da3 \
    --da3-model-dir "$DA3_MODEL_DIR" \
    ${DA3_REPO_ROOT:+--da3-repo-root "$DA3_REPO_ROOT"} \
    --use-reference-model \
    --fk-urdf "$FK_URDF" \
    --fk-weight "$FK_WEIGHT" \
    --fk-ws-weight "$FK_WS_WEIGHT" \
    --fk-singularity-weight "$FK_SING_WEIGHT" \
    --fk-ee-vel-weight "$FK_EEVEL_WEIGHT" \
    --fk-ee-acc-weight "$FK_EEACC_WEIGHT" \
    --fk-chain-weight "$FK_CHAIN_WEIGHT" \
    --fk-dual-arm-weight "$FK_DUAL_WEIGHT" \
    --group-size "$GROUP_SIZE" \
    --horizon-steps "$HORIZON" \
    --hist-len "$HIST_LEN" \
    --steps "$GRPO_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --override-n-frames "$N_FRAMES" \
    --override-height "$HEIGHT" \
    --override-width "$WIDTH" \
    --override-sample-steps "$SAMPLE_STEPS" \
    --lr "$LR" \
    --num-inner-epochs "$NUM_INNER_EPOCHS" \
    --clip-eps "$CLIP_EPS" \
    --beta-kl "$BETA_KL" \
    --surrogate-sigma "$SURROGATE_SIGMA" \
    --log-ratio-clip "$LOG_RATIO_CLIP" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --ref-update-interval "$REF_UPDATE_INTERVAL" \
    --discount-gamma "$DISCOUNT_GAMMA" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --wan-device "$WAN_DEVICE" \
    --ref-device "$REF_DEVICE" \
    --depth-device "$DEPTH_DEVICE" \
    --idm-device "$IDM_DEVICE"
