#!/usr/bin/env bash
set -Eeuo pipefail

# 减少显存碎片，避免因碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${DEBUG_SHELL:-0}" == "1" ]]; then
  set -x
fi

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2' ERR

# ============================================================================
# EVA Wan-GRPO 正式完整训练脚本
# ============================================================================
#
# 与 smoke test 脚本的区别：
#   1. 不使用 --override-height/width/n-frames/sample-steps（使用 YAML 中的完整分辨率 480x640, 49帧）
#   2. 训练步数更多（默认 1000 步）
#   3. 超参数为正式训练推荐值
#
# 使用方式：
#   conda activate <your_env>
#   bash scripts/run_wan_grpo_full_train.sh
#
# 可选覆盖（环境变量）：
#   STEPS=2000 GROUP_SIZE=2 LR=5e-6 bash scripts/run_wan_grpo_full_train.sh
#
# GPU 配置：
#   单卡模式（默认）：所有模型放 cuda:0
#   双卡模式：设置 USE_REF=1 REF_DEVICE=cuda:1
#
# ============================================================================

# ---- 环境变量 ----
export PYTHONPATH="/root/autodl-tmp/dpy/myrepos/EVA-main:/root/autodl-tmp/dpy/myrepos/Depth-Anything-3-main/src:/root/autodl-tmp/dpy/myrepos/vidar:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# ---- 路径配置 ----
REPO_ROOT="/root/autodl-tmp/dpy/myrepos/EVA-main"
CONFIG_PATH="/root/autodl-tmp/dpy/myrepos/EVA-main/configurations/algorithm/wan_i2v.yaml"
CONDITION_BANK="${CONDITION_BANK:-/root/autodl-tmp/dpy/wan_grpo_condition_bank.pt}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/dpy/grpo_full_train}"

# ---- 模型路径 ----
VIDAR_CKPT="${VIDAR_CKPT:-/root/autodl-tmp/dpy/my_models/vidar/152000.pt}"
DINO_DIR="${DINO_DIR:-/root/autodl-tmp/dpy/my_models/dinov2-with-registers-base}"
DA3_DIR="${DA3_DIR:-/root/autodl-tmp/dpy/my_models/DA3-LARGE-1.1}"
DA3_REPO_ROOT="${DA3_REPO_ROOT:-/root/autodl-tmp/dpy/myrepos/Depth-Anything-3-main}"

# ---- GPU 设备 ----
WAN_DEVICE="${WAN_DEVICE:-cuda:0}"
DEPTH_DEVICE="${DEPTH_DEVICE:-cuda:1}"
IDM_DEVICE="${IDM_DEVICE:-cuda:1}"
REF_DEVICE="${REF_DEVICE:-cuda:1}"

# ---- 训练超参数 ----
STEPS="${STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
GROUP_SIZE="${GROUP_SIZE:-8}"
HORIZON_STEPS="${HORIZON_STEPS:-2}"
NUM_INNER_EPOCHS="${NUM_INNER_EPOCHS:-4}"
HIST_LEN="${HIST_LEN:-1}"
FLOW_SAMPLING_NOISE_STD="${FLOW_SAMPLING_NOISE_STD:-0.08}"
DISCOUNT_GAMMA="${DISCOUNT_GAMMA:-0.98}"

# ---- 优化器 ----
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
CLIP_EPS="${CLIP_EPS:-0.001}"
BETA_KL="${BETA_KL:-0.004}"
SURROGATE_SIGMA="${SURROGATE_SIGMA:-1.0}"
LOG_RATIO_CLIP="${LOG_RATIO_CLIP:-5.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
REF_UPDATE_INTERVAL="${REF_UPDATE_INTERVAL:-0}"

# ---- LoRA ----
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
GRADIENT_CHECKPOINTING_RATE="${GRADIENT_CHECKPOINTING_RATE:-1.0}"

# ---- 分辨率覆盖（适配 96GB 单卡）----
# 14B 模型在全分辨率 (480x640x49帧) 下单次 forward+backward 就会超 96GB
# 320x480x17帧 是论文推荐的训练分辨率，显存约 40-50GB，足够 96GB 安全运行
OVERRIDE_HEIGHT="${OVERRIDE_HEIGHT:-320}"
OVERRIDE_WIDTH="${OVERRIDE_WIDTH:-480}"
OVERRIDE_N_FRAMES="${OVERRIDE_N_FRAMES:-17}"
OVERRIDE_SAMPLE_STEPS="${OVERRIDE_SAMPLE_STEPS:-10}"

# ---- Reward 权重 ----
HARD_VETO_PENALTY="${HARD_VETO_PENALTY:-50.0}"
# control_state_t and IDM outputs are both normalized [0,1].
# 0.25 = allow 25% range change per step (tight for fast motions).
# Set to 0.5 as a relaxed starting point; tune down after confirming reward flows.
MAX_CONTROL_DELTA="${MAX_CONTROL_DELTA:-0.25}"
FEASIBILITY_WEIGHT="${FEASIBILITY_WEIGHT:-1.0}"
ACTION_RECOVERY_WEIGHT="${ACTION_RECOVERY_WEIGHT:-1.0}"
IDM_STABILITY_WEIGHT="${IDM_STABILITY_WEIGHT:-0.1}"
DOF_WEIGHTS="${DOF_WEIGHTS:-1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2}"

# ============================================================================
# 构建命令
# ============================================================================
cd "${REPO_ROOT}"

# 动态检测 GPU 模式：收集所有唯一的设备，判断单卡还是多卡
_DEVICES_USED="${WAN_DEVICE}"
for _d in "${DEPTH_DEVICE}" "${IDM_DEVICE}" "${REF_DEVICE:-}"; do
  if [[ -n "${_d}" && "${_DEVICES_USED}" != *"${_d}"* ]]; then
    _DEVICES_USED="${_DEVICES_USED}, ${_d}"
  fi
done
if [[ "${_DEVICES_USED}" == *","* ]]; then
  _GPU_MODE="多卡"
else
  _GPU_MODE="单卡"
fi

echo "============================================"
echo " EVA Wan-GRPO RL Post-Training (${_GPU_MODE})"
echo "============================================"
echo " Config:         ${CONFIG_PATH}"
echo " Condition Bank: ${CONDITION_BANK}"
echo " Save Dir:       ${SAVE_DIR}"
echo " Steps:          ${STEPS}"
echo " Group Size:     ${GROUP_SIZE}"
echo " Horizon Steps:  ${HORIZON_STEPS}"
echo " LR:             ${LR}"
echo " Resolution:     ${OVERRIDE_HEIGHT}x${OVERRIDE_WIDTH}, ${OVERRIDE_N_FRAMES} frames, ${OVERRIDE_SAMPLE_STEPS} denoise steps"
echo " Inner Epochs:   ${NUM_INNER_EPOCHS}"
echo " Discount Gamma: ${DISCOUNT_GAMMA}"
echo " --- GPU 分配 ---"
echo " WAN Model:      ${WAN_DEVICE}"
echo " Depth Model:    ${DEPTH_DEVICE}"
echo " IDM Model:      ${IDM_DEVICE}"
[[ -n "${REF_DEVICE:-}" ]] && echo " Ref Model:      ${REF_DEVICE}"
echo " 模式:           ${_GPU_MODE}（使用设备: ${_DEVICES_USED}）"
echo "============================================"

python -m algorithms.wan.run_state_unrolled_grpo \
  --config "${CONFIG_PATH}" \
  --condition-bank "${CONDITION_BANK}" \
  --save-dir "${SAVE_DIR}" \
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
  --use-reference-model \
  --idm-checkpoint "${VIDAR_CKPT}" \
  --idm-backend vidar \
  --idm-model-name mask \
  --idm-model-output-dim 16 \
  --depth-backend da3 \
  --da3-model-dir "${DA3_DIR}" \
  --da3-repo-root "${DA3_REPO_ROOT}" \
  --wan-device "${WAN_DEVICE}" \
  --depth-device "${DEPTH_DEVICE}" \
  --idm-device "${IDM_DEVICE}" \
  --ref-device "${REF_DEVICE}" \
  --disable-model-compile \
  --disable-clip-compile \
  --disable-vae-compile \
  --disable-text-encoder-compile \
  ${OVERRIDE_HEIGHT:+--override-height "${OVERRIDE_HEIGHT}"} \
  ${OVERRIDE_WIDTH:+--override-width "${OVERRIDE_WIDTH}"} \
  ${OVERRIDE_N_FRAMES:+--override-n-frames "${OVERRIDE_N_FRAMES}"} \
  ${OVERRIDE_SAMPLE_STEPS:+--override-sample-steps "${OVERRIDE_SAMPLE_STEPS}"} \
  --hard-veto-penalty "${HARD_VETO_PENALTY}" \
  --max-control-delta "${MAX_CONTROL_DELTA}" \
  --feasibility-weight "${FEASIBILITY_WEIGHT}" \
  --action-recovery-weight "${ACTION_RECOVERY_WEIGHT}" \
  --idm-stability-weight "${IDM_STABILITY_WEIGHT}" \
  --dof-weights "${DOF_WEIGHTS}"
