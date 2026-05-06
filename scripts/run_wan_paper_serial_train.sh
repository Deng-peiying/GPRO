#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${DEBUG_SHELL:-0}" == "1" ]]; then
  set -x
fi

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2' ERR

# Temporary bring-up wrapper.
# The paper's SFT-tuned weights are not ready yet, so this uses the EVA
# original checkpoint as the starting point for FK-grounded GRPO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TWO_STAGE_CKPT="${TWO_STAGE_CKPT:-/root/autodl-tmp/main_exp/EVA_Wan/eva_i2v_14B.ckpt}"
CONDITION_BANK="${CONDITION_BANK:-/root/autodl-tmp/main_exp/wan_grpo_condition_bank_triple.pt}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/dpy/grpo_serial_train}"

IDM_CHECKPOINT="${IDM_CHECKPOINT:-/root/autodl-tmp/main_exp/commit-freeze_test/output/full/152000.pt}"
VIDAR_REPO_ROOT="${VIDAR_REPO_ROOT:-/root/autodl-tmp/main_exp/commit-freeze_test}"
FK_URDF="${FK_URDF:-/root/autodl-tmp/main_exp/RoboTwin/assets/embodiments/franka-panda/panda.urdf}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-/root/autodl-tmp/main_exp/Depth-Anything-3-main/model/DA3-LARGE-1.1}"
DA3_REPO_ROOT="${DA3_REPO_ROOT:-/root/autodl-tmp/main_exp/Depth-Anything-3-main}"
export PYTHONPATH="${PROJECT_ROOT}:${DA3_REPO_ROOT}/src:${VIDAR_REPO_ROOT}:${PYTHONPATH:-}"

WAN_DEVICE="${WAN_DEVICE:-cuda:0}"
REF_DEVICE="${REF_DEVICE:-cuda:1}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda:0}"
IDM_DEVICE="${IDM_DEVICE:-cuda:0}"

STEPS="${STEPS:-3}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
GROUP_SIZE="${GROUP_SIZE:-1}"
HORIZON_STEPS="${HORIZON_STEPS:-1}"
NUM_INNER_EPOCHS="${NUM_INNER_EPOCHS:-1}"
HIST_LEN="${HIST_LEN:-1}"
FLOW_SAMPLING_NOISE_STD="${FLOW_SAMPLING_NOISE_STD:-0.05}"
DISCOUNT_GAMMA="${DISCOUNT_GAMMA:-1.0}"

LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP_EPS="${CLIP_EPS:-0.1}"
BETA_KL="${BETA_KL:-1e-3}"
SURROGATE_SIGMA="${SURROGATE_SIGMA:-1.0}"
LOG_RATIO_CLIP="${LOG_RATIO_CLIP:-2.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
REF_UPDATE_INTERVAL="${REF_UPDATE_INTERVAL:-0}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
GRADIENT_CHECKPOINTING_RATE="${GRADIENT_CHECKPOINTING_RATE:-1.0}"

OVERRIDE_HEIGHT="${OVERRIDE_HEIGHT:-720}"
OVERRIDE_WIDTH="${OVERRIDE_WIDTH:-640}"
OVERRIDE_N_FRAMES="${OVERRIDE_N_FRAMES:-49}"
OVERRIDE_SAMPLE_STEPS="${OVERRIDE_SAMPLE_STEPS:-12}"

HARD_VETO_PENALTY="${HARD_VETO_PENALTY:-50.0}"
MAX_CONTROL_DELTA="${MAX_CONTROL_DELTA:-0.25}"
FEASIBILITY_WEIGHT="${FEASIBILITY_WEIGHT:-1.0}"
ACTION_RECOVERY_WEIGHT="${ACTION_RECOVERY_WEIGHT:-1.0}"
IDM_STABILITY_WEIGHT="${IDM_STABILITY_WEIGHT:-0.1}"
DOF_WEIGHTS="${DOF_WEIGHTS:-1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}"
FK_WEIGHT="${FK_WEIGHT:-0.5}"
FK_WS_WEIGHT="${FK_WS_WEIGHT:-1.0}"
FK_SING_WEIGHT="${FK_SING_WEIGHT:-0.5}"
FK_EEVEL_WEIGHT="${FK_EEVEL_WEIGHT:-0.3}"
FK_EEACC_WEIGHT="${FK_EEACC_WEIGHT:-0.2}"
FK_CHAIN_WEIGHT="${FK_CHAIN_WEIGHT:-1.0}"
FK_DUAL_WEIGHT="${FK_DUAL_WEIGHT:-0.3}"

cd "${PROJECT_ROOT}"

python -m algorithms.wan.run_state_unrolled_grpo \
  --config "${PROJECT_ROOT}/configurations/algorithm/wan_i2v.yaml" \
  --condition-bank "${CONDITION_BANK}" \
  --save-dir "${SAVE_DIR}" \
  --tuned-ckpt "${TWO_STAGE_CKPT}" \
  --steps "${STEPS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --group-size "${GROUP_SIZE}" \
  --horizon-steps "${HORIZON_STEPS}" \
  --num-inner-epochs "${NUM_INNER_EPOCHS}" \
  --hist-len "${HIST_LEN}" \
  --flow-sampling-noise-std "${FLOW_SAMPLING_NOISE_STD}" \
  --discount-gamma "${DISCOUNT_GAMMA}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --clip-eps "${CLIP_EPS}" \
  --beta-kl "${BETA_KL}" \
  --surrogate-sigma "${SURROGATE_SIGMA}" \
  --log-ratio-clip "${LOG_RATIO_CLIP}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --ref-update-interval "${REF_UPDATE_INTERVAL}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --gradient-checkpointing-rate "${GRADIENT_CHECKPOINTING_RATE}" \
  --idm-checkpoint "${IDM_CHECKPOINT}" \
  --idm-backend vidar \
  --idm-model-name mask \
  --idm-model-output-dim 16 \
  --depth-backend da3 \
  --da3-model-dir "${DA3_MODEL_DIR}" \
  --da3-repo-root "${DA3_REPO_ROOT}" \
  --fk-urdf "${FK_URDF}" \
  --fk-weight "${FK_WEIGHT}" \
  --fk-ws-weight "${FK_WS_WEIGHT}" \
  --fk-singularity-weight "${FK_SING_WEIGHT}" \
  --fk-ee-vel-weight "${FK_EEVEL_WEIGHT}" \
  --fk-ee-acc-weight "${FK_EEACC_WEIGHT}" \
  --fk-chain-weight "${FK_CHAIN_WEIGHT}" \
  --fk-dual-arm-weight "${FK_DUAL_WEIGHT}" \
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
