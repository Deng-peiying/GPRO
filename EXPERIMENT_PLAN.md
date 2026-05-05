# EVA 后训练实验计划

## 实验总览

```
SFT 三级消融:
  A: Vanilla SFT ─────────→ 随机窗口，单阶段（EVA基线）
  B: Entry-focused SFT ───→ arm-entry窗口，单阶段
  C: Two-stage SFT ───────→ B → full-horizon 两阶段

完整管线:
  D: Two-stage SFT + FK GRPO → C的ckpt → FK 6-component reward

依赖关系: A独立 | B独立 | C依赖B | D依赖C
并行策略: A和B可同时跑，节省1天
```

---

# Part 1: GRPO 验证（最高优先级）

## 目的

确认 FK-grounded GRPO 全链路可运行：IDM → DA3 depth → rollout → FK reward → 梯度更新。

**这步不通，后续所有实验都无法开展。**

## 需要做的事

### 1.1 环境检查

确认服务器上所有依赖已安装：

```bash
conda activate eva
pip show numpy moviepy addict torch  # numpy>=2.0, moviepy==1.0.3
```

如果缺包：
```bash
pip install "numpy>=2.0" moviepy==1.0.3 addict
```

### 1.2 确认关键文件存在

- Wan2.1 I2V 14B base model: `/home/zhangmohan/models/Wan2.1-I2V-14B-480P/`
- IDM checkpoint: 你的 IDM 模型的 `.pt` 或 `.ckpt` 文件
- DA3 checkpoint: Depth-Anything-3 模型
- Condition bank: `wan_grpo_condition_bank.pt`（如果没有，需用 `scripts/build_wan_grpo_condition_bank.py` 构建）
- Franka URDF: `panda_arm.urdf`（可选，验证 FK 时用）

### 1.3 运行 Smoke Test

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

python -m algorithms.wan.run_state_unrolled_grpo \
    --config configurations/algorithm/wan_i2v.yaml \
    --condition-bank /path/to/wan_grpo_condition_bank.pt \
    --idm-backend vidar \
    --idm-checkpoint /path/to/idm_checkpoint.pt \
    --da3-checkpoint /path/to/da3_checkpoint \
    --tuned-ckpt /path/to/two_stage_sft.ckpt \
    --use-reference-model \
    --use-fk-reward \
    --fk-urdf /path/to/franka_panda.urdf \
    --group-size 2 \
    --horizon-steps 2 \
    --grpo-steps 5 \
    --wandb-mode disabled
```

关键参数说明：
- `--tuned-ckpt`：先用任意 Wan2.1 LoRA checkpoint（没有的话暂时去掉这个参数，用 base model）
- `--group-size 2 --horizon-steps 2`：最小配置，验证链路
- `--grpo-steps 5`：只跑 5 步，确认 loss 正常下降
- `--wandb-mode disabled`：不连 wandb

### 1.4 验证通过标准

- [ ] 无 import error / ModuleNotFoundError
- [ ] 无 OOM（如果 OOM，尝试 `--group-size 1 --horizon-steps 1`）
- [ ] rollout 正常完成（log 中出现 reward 值）
- [ ] loss 正常下降（5 步内 loss 不 NaN、不爆炸）
- [ ] 如果有 FK URDF：FK reward 6 个组件都有非零值输出（workspace, singularity, ee_vel, ee_acc, fk_chain, dual_arm）

### 1.5 如果失败

| 症状 | 可能原因 | 修复 |
|------|----------|------|
| OOM | deepcopy reference model | 去掉 `--use-reference-model`，或先 move 到 CPU |
| FK reward 全是 0 | FK 输入 action 范围不对 | 检查 `action_t` 的 scale（应 ≈ joint angles in radians） |
| IDM 输出 NaN | IDM checkpoint 不兼容 | 确认 IDM 输入输出维度 16，确认 normalization |
| rollout 报错缺少 key | condition_bank 格式不匹配 | 用 `build_wan_grpo_condition_bank.py` 重新构建 |

---

# Part 2: SFT 消融实验

## 目的

证明 two-stage SFT（entry-focused → full-horizon）优于 vanilla SFT，在 embodiment-level 指标上有逐级提升。

## 2.0 数据准备（一次性）

### 2.0.1 确认输入数据

- HDF5 目录：`/path/to/hdf5_episodes/`（含 `episode*.hdf5`）
- Metadata JSON：`/path/to/episode_meta.json`（含 `source_task` 和 `task_info`）

### 2.0.2 生成两份数据集

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

# Vanilla 数据集：每个 episode 一个完整视频 (composite 3-view)
python scripts/prepare_robotwin_sft_data.py \
    --data-dir /path/to/hdf5_episodes \
    --meta-json /path/to/episode_meta.json \
    --output-dir /home/zhangmohan/data/robotwin_sft_vanilla \
    --mode vanilla --composite --fps 16

# Arm-entry 数据集：只保留 arm 入场窗口 (composite 3-view)
python scripts/prepare_robotwin_sft_data.py \
    --data-dir /path/to/hdf5_episodes \
    --meta-json /path/to/episode_meta.json \
    --output-dir /home/zhangmohan/data/robotwin_sft_entry \
    --mode arm-entry --composite \
    --n-frames 17 \
    --arm-delta-thresh 0.3 \
    --window-stride 8 \
    --fps 16
```

输出结构 (composite 640×720):
```
data/robotwin_sft_vanilla/
    metadata.csv
    videos/episode_000000.mp4 ...  (640×720, 3-view composite)

data/robotwin_sft_entry/
    metadata.csv
    videos/episode_000000_w0000.mp4 ...  (每个 arm-entry 窗口一个视频)
```

### 2.0.3 检查数据

```bash
# 确认 CSV 行数
wc -l /home/zhangmohan/data/robotwin_sft_vanilla/metadata.csv
wc -l /home/zhangmohan/data/robotwin_sft_entry/metadata.csv

# 确认视频可读 (640×720 composite)
python -c "
import cv2
cap = cv2.VideoCapture('/home/zhangmohan/data/robotwin_sft_vanilla/videos/episode_000000.mp4')
print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'Size:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
"
```

## 2.A: Vanilla SFT（基线）

### 运行

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

DATA_ROOT=/home/zhangmohan/data/robotwin_sft_vanilla \
WAN_CKPT=/home/zhangmohan/models/Wan2.1-I2V-14B-480P \
bash scripts/run_wan_sft_vanilla.sh
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_frames | 17 | ~1s @16fps |
| LoRA rank | 32 | |
| LR | 5e-6 | |
| batch_size | 2 (per GPU) × 8 GPU × 4 accum = 64 effective | |
| max_steps | 50000 | |
| 预计时间 | ~12h | |

### 产出

- `vanilla.ckpt`：最后一步 checkpoint
- wandb log：train/loss 曲线

## 2.B: Entry-focused SFT

### 运行

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

DATA_ROOT=/home/zhangmohan/data/robotwin_sft_entry \
WAN_CKPT=/home/zhangmohan/models/Wan2.1-I2V-14B-480P \
bash scripts/run_wan_sft_entry.sh
```

### 关键参数

| 参数 | 值 |
|------|-----|
| n_frames | 17 |
| LoRA rank | 32 |
| LR | 5e-6 |
| max_steps | 50000 |
| 预计时间 | ~12h（可和A并行） |

### 产出

- `entry.ckpt`

## 2.C: Two-stage SFT

### 前置条件

B 完成后，取 loss 最低的 checkpoint。

### 运行

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

ENTRY_CKPT=/path/to/entry.ckpt \
DATA_ROOT=/home/zhangmohan/data/robotwin_sft_vanilla \
WAN_CKPT=/home/zhangmohan/models/Wan2.1-I2V-14B-480P \
bash scripts/run_wan_sft_two_stage.sh
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_frames | 49 | full horizon |
| LR | 3e-6 | 比 Stage1 低 |
| batch_size | 1 × 8 × 8 = 64 effective | 每卡 batch=1（49帧更长） |
| max_steps | 50000 | |
| 预计时间 | ~18h | |

### 产出

- `two_stage.ckpt`

## 2.D: Two-stage SFT + FK GRPO

### 前置条件

- Part 1 的 smoke test 已通过
- C 完成

### 运行

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

TWO_STAGE_CKPT=/path/to/two_stage.ckpt \
CONDITION_BANK=/path/to/wan_grpo_condition_bank.pt \
IDM_CHECKPOINT=/path/to/idm_checkpoint.pt \
DA3_CHECKPOINT=/path/to/da3_checkpoint \
FK_URDF=/path/to/franka_panda.urdf \
bash scripts/run_wan_fk_grpo.sh
```

### 关键参数

| 参数 | 值 |
|------|-----|
| group_size | 4 |
| horizon_steps | 4 |
| GRPO steps | 8000 |
| LR | 1e-6 |
| FK weight | 0.5 |
| 预计时间 | ~24h |

### 产出

- `fk_grpo.ckpt`

## 2.E: 统一评估

### 运行

```bash
cd /home/zhangmohan/workspace/code/EVA-main_v0.5

python scripts/eval_embodiment_metrics.py \
    --test-data /path/to/test_hdf5_episodes \
    --checkpoints \
        /path/to/vanilla.ckpt \
        /path/to/entry.ckpt \
        /path/to/two_stage.ckpt \
        /path/to/fk_grpo.ckpt \
    --labels "Vanilla SFT" "Entry SFT" "Two-stage SFT" "+ FK GRPO" \
    --output results.json \
    --table-output results_table.tex
```

### 产出

- `results.json`：所有指标
- `results_table.tex`：LaTeX 表格，可直接插入论文

---

## 时间线总览

```
         Day1    Day2    Day3    Day4    Day5    Day6    Day7
Part 1:  [验证]
Part 2:
  Data   [准备]
  A/B    [====并行====]
  C                    [==C==]
  D                            [===D===]
  Eval                                   [评估]
  Paper                                          [buffer]
```

## 分工建议

| 角色 | 负责任务 |
|------|----------|
| 你 | Part 1 (GRPO验证) + Part 2.D (GRPO训练) |
| 合作者 | Part 2.0 (数据准备) + Part 2.A-C (SFT训练) + Part 2.E (评估) |

---

## 需要提前确认的信息

- [ ] 服务器 HDF5 数据路径
- [ ] episode metadata JSON 路径和格式
- [ ] IDM checkpoint 路径
- [ ] DA3 checkpoint 路径
- [ ] Condition bank 路径（或是否需要新建）
- [ ] Franka URDF 路径
