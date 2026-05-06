#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${DEBUG_SHELL:-0}" == "1" ]]; then
  set -x
fi

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configurations/algorithm/wan_i2v.yaml}"
CONDITION_BANK="${CONDITION_BANK:-/root/autodl-tmp/main_exp/wan_grpo_condition_bank_triple.pt}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/dpy/grpo_ddp_2worker_49f}"
TWO_STAGE_CKPT="${TWO_STAGE_CKPT:-/root/autodl-tmp/main_exp/EVA_Wan/eva_i2v_14B.ckpt}"

VIDAR_CKPT="${VIDAR_CKPT:-/root/autodl-tmp/main_exp/commit-freeze_test/output/full/152000.pt}"
VIDAR_REPO_ROOT="${VIDAR_REPO_ROOT:-/root/autodl-tmp/main_exp/commit-freeze_test}"
FK_URDF="${FK_URDF:-/root/autodl-tmp/main_exp/RoboTwin/assets/embodiments/franka-panda/panda.urdf}"
DA3_DIR="${DA3_DIR:-/root/autodl-tmp/main_exp/Depth-Anything-3-main/model/DA3-LARGE-1.1}"
DA3_REPO_ROOT="${DA3_REPO_ROOT:-/root/autodl-tmp/main_exp/Depth-Anything-3-main}"
export PYTHONPATH="${PROJECT_ROOT}:${DA3_REPO_ROOT}/src:${VIDAR_REPO_ROOT}:${PYTHONPATH:-}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"

# Rank-local device mapping.
# Worker 0: WAN on 0, depth/text/vae on 1, IDM on 2
# Worker 1: WAN on 3, depth/text/vae on 4, IDM on 5
WAN_DEVICES="${WAN_DEVICES:-cuda:0,cuda:3}"
DEPTH_DEVICES="${DEPTH_DEVICES:-cuda:1,cuda:4}"
IDM_DEVICES="${IDM_DEVICES:-cuda:2,cuda:5}"

LOCAL_GROUP_SIZE="${GROUP_SIZE:-1}"
STEPS="${STEPS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
HORIZON_STEPS="${HORIZON_STEPS:-2}"
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

OVERRIDE_HEIGHT="${OVERRIDE_HEIGHT:-192}"
OVERRIDE_WIDTH="${OVERRIDE_WIDTH:-256}"
OVERRIDE_N_FRAMES="${OVERRIDE_N_FRAMES:-49}"
OVERRIDE_SAMPLE_STEPS="${OVERRIDE_SAMPLE_STEPS:-4}"

USE_REFERENCE_MODEL="${USE_REFERENCE_MODEL:-0}"
STAGGER_DISTRIBUTED_LOAD="${STAGGER_DISTRIBUTED_LOAD:-1}"
RANK0_CONDITION_BANK="${RANK0_CONDITION_BANK:-1}"
HARD_VETO_PENALTY="${HARD_VETO_PENALTY:-50.0}"
MAX_CONTROL_DELTA="${MAX_CONTROL_DELTA:-0.5}"
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

torchrun \
  --nnodes=1 \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  -m algorithms.wan.run_state_unrolled_grpo \
  --config "${CONFIG_PATH}" \
  --condition-bank "${CONDITION_BANK}" \
  --save-dir "${SAVE_DIR}" \
  --tuned-ckpt "${TWO_STAGE_CKPT}" \
  --steps "${STEPS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --group-size "${LOCAL_GROUP_SIZE}" \
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
  --idm-checkpoint "${VIDAR_CKPT}" \
  --idm-backend vidar \
  --idm-model-name mask \
  --idm-model-output-dim 16 \
  --depth-backend da3 \
  --da3-model-dir "${DA3_DIR}" \
  --da3-repo-root "${DA3_REPO_ROOT}" \
  --fk-urdf "${FK_URDF}" \
  --fk-weight "${FK_WEIGHT}" \
  --fk-ws-weight "${FK_WS_WEIGHT}" \
  --fk-singularity-weight "${FK_SING_WEIGHT}" \
  --fk-ee-vel-weight "${FK_EEVEL_WEIGHT}" \
  --fk-ee-acc-weight "${FK_EEACC_WEIGHT}" \
  --fk-chain-weight "${FK_CHAIN_WEIGHT}" \
  --fk-dual-arm-weight "${FK_DUAL_WEIGHT}" \
  --wan-device "${WAN_DEVICES}" \
  --depth-device "${DEPTH_DEVICES}" \
  --idm-device "${IDM_DEVICES}" \
  --override-height "${OVERRIDE_HEIGHT}" \
  --override-width "${OVERRIDE_WIDTH}" \
  --override-n-frames "${OVERRIDE_N_FRAMES}" \
  --override-sample-steps "${OVERRIDE_SAMPLE_STEPS}" \
  --disable-model-compile \
  --disable-clip-compile \
  --disable-vae-compile \
  --disable-text-encoder-compile \
  $( [[ "${STAGGER_DISTRIBUTED_LOAD}" == "1" ]] && printf '%s' "--stagger-distributed-load" ) \
  $( [[ "${RANK0_CONDITION_BANK}" == "1" ]] && printf '%s' "--rank0-condition-bank" ) \
  --hard-veto-penalty "${HARD_VETO_PENALTY}" \
  --max-control-delta "${MAX_CONTROL_DELTA}" \
  --feasibility-weight "${FEASIBILITY_WEIGHT}" \
  --action-recovery-weight "${ACTION_RECOVERY_WEIGHT}" \
  --idm-stability-weight "${IDM_STABILITY_WEIGHT}" \
  --dof-weights "${DOF_WEIGHTS}" \
  $( [[ "${USE_REFERENCE_MODEL}" == "1" ]] && printf '%s' "--use-reference-model" )
