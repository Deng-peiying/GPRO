from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

from .depth_backends import build_depth_backend
from .idm_backends import build_idm_backend
from .run_single_step_replan import adapt_condition_batch
from .state_action_reward import StepExecutabilityReward
from .state_unrolled_grpo_trainer import StateUnrolledGRPOTrainer, StateUnrolledGRPOTrainerConfig
from .wan_i2v import WanImageToVideo


def load_config(config_path: str):
    config_file = Path(config_path).resolve()
    repo_root = config_file.parents[2]
    root_cfg = OmegaConf.create({})

    def _merge_if_exists(path: Path):
        nonlocal root_cfg
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                root_cfg = OmegaConf.merge(root_cfg, OmegaConf.create(yaml.safe_load(f)))

    _merge_if_exists(repo_root / "configurations" / "config.yaml")

    experiment_cfg = OmegaConf.create({})
    dataset_cfg = OmegaConf.create({})
    algorithm_cfg = OmegaConf.create({})

    for path in (
        repo_root / "configurations" / "experiment" / "base_experiment.yaml",
        repo_root / "configurations" / "experiment" / "base_pytorch.yaml",
        repo_root / "configurations" / "experiment" / "exp_inference.yaml",
    ):
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                experiment_cfg = OmegaConf.merge(experiment_cfg, OmegaConf.create(yaml.safe_load(f)))

    for path in (
        repo_root / "configurations" / "dataset" / "video_base.yaml",
        repo_root / "configurations" / "dataset" / "image_csv.yaml",
    ):
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                dataset_cfg = OmegaConf.merge(dataset_cfg, OmegaConf.create(yaml.safe_load(f)))

    if config_file.name == "wan_i2v.yaml":
        with (config_file.parent / "wan_t2v.yaml").open("r", encoding="utf-8") as f:
            algorithm_cfg = OmegaConf.merge(algorithm_cfg, OmegaConf.create(yaml.safe_load(f)))
    with config_file.open("r", encoding="utf-8") as f:
        algorithm_cfg = OmegaConf.merge(algorithm_cfg, OmegaConf.create(yaml.safe_load(f)))

    root_cfg.experiment = experiment_cfg
    root_cfg.dataset = dataset_cfg
    root_cfg.algorithm = algorithm_cfg
    root_cfg.algorithm.debug = root_cfg.get("debug", False)

    resolved = OmegaConf.create(OmegaConf.to_container(root_cfg, resolve=True))
    return resolved.algorithm


def build_algo(cfg):
    with open_dict(cfg):
        cfg.force_training = True
    algo = WanImageToVideo(cfg)
    algo.configure_model()
    algo.eval()
    return algo


def apply_runtime_overrides(cfg, args):
    with open_dict(cfg):
        if args.override_height is not None:
            cfg.height = int(args.override_height)
        if args.override_width is not None:
            cfg.width = int(args.override_width)
        if args.override_n_frames is not None:
            cfg.n_frames = int(args.override_n_frames)
        if args.override_sample_steps is not None:
            cfg.sample_steps = int(args.override_sample_steps)
        cfg.force_training = True
        cfg.model.use_lora = True
        cfg.model.lora_rank = int(args.lora_rank)
        cfg.model.lora_alpha = int(args.lora_alpha)
        cfg.model.lora_dropout = float(args.lora_dropout)
        if args.disable_model_compile:
            cfg.model.compile = False
        if cfg.get("clip") is not None and args.disable_clip_compile:
            cfg.clip.compile = False
        if cfg.get("vae") is not None and args.disable_vae_compile:
            cfg.vae.compile = False
        if cfg.get("text_encoder") is not None and args.disable_text_encoder_compile:
            cfg.text_encoder.compile = False
        if args.gradient_checkpointing_rate is not None:
            cfg.gradient_checkpointing_rate = float(args.gradient_checkpointing_rate)
    return cfg


def build_optimizer(algo, lr: float, weight_decay: float):
    params = [p for p in algo.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def load_condition_bank(path: str):
    bank_path = Path(path)
    if bank_path.suffix == ".pt":
        return torch.load(bank_path)
    raise ValueError(f"Unsupported condition bank format: {bank_path}")


def get_condition_batch(condition_bank, index: int, device: torch.device, *, n_frames: int, height: int, width: int):
    item = condition_bank[index]
    cond_batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            cond_batch[key] = value.to(device)
        else:
            cond_batch[key] = value
    return adapt_condition_batch(cond_batch, n_frames=n_frames, height=height, width=width)


def parse_optional_float_list(raw: str | None) -> torch.Tensor | None:
    if raw is None:
        return None
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        return None
    return torch.tensor(values, dtype=torch.float32)


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def validate_args(args) -> None:
    positive_int_fields = {
        "steps": args.steps,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "group_size": args.group_size,
        "horizon_steps": args.horizon_steps,
        "hist_len": args.hist_len,
        "idm_model_output_dim": args.idm_model_output_dim,
    }
    for name, value in positive_int_fields.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0, got {value}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be >= 0, got {args.weight_decay}")
    if args.surrogate_sigma <= 0:
        raise ValueError(f"--surrogate-sigma must be > 0, got {args.surrogate_sigma}")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        raise ValueError(f"--grad-clip-norm must be > 0 when set, got {args.grad_clip_norm}")
    if args.ref_update_interval < 0:
        raise ValueError(f"--ref-update-interval must be >= 0, got {args.ref_update_interval}")


def _move_state_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _move_state_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_state_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_state_to_cpu(value) for value in obj)
    return obj


def extract_trainable_state_dict(module) -> dict[str, torch.Tensor]:
    state_dict = module.state_dict()
    trainable_names = {
        name
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    filtered = {}
    for key, value in state_dict.items():
        if any(key == name or key.startswith(f"{name}.") for name in trainable_names):
            filtered[key] = value.detach().cpu()
    return filtered


def main():
    parser = argparse.ArgumentParser(description="State-unrolled multi-step GRPO runner for Wan + DA3 + IDM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition-bank", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--hist-len", type=int, default=1)
    parser.add_argument("--flow-sampling-noise-std", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--surrogate-sigma", type=float, default=1.0)
    parser.add_argument("--log-ratio-clip", type=float, default=5.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ref-update-interval", type=int, default=0)
    parser.add_argument("--discount-gamma", type=float, default=1.0,
                        help="Exponential discount factor for multi-step reward: R = sum(gamma^t * r_t). "
                             "1.0 = no discount (default), 0.99 = mild decay, 0.9 = strong decay on later steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-inner-epochs", type=int, default=1,
                        help="Number of times to update policy per rollout batch (PPO/GRPO style).")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing-rate", type=float, default=1.0)
    parser.add_argument("--use-reference-model", action="store_true")
    parser.add_argument("--idm-checkpoint", required=True)
    parser.add_argument("--idm-backend", choices=["vidar", "custom"], default="vidar")
    parser.add_argument("--idm-model-name", default="mask")
    parser.add_argument("--idm-dinov2-name", default="")
    parser.add_argument("--idm-left-arm-dim", type=int, default=7)
    parser.add_argument("--idm-right-arm-dim", type=int, default=7)
    parser.add_argument("--idm-model-output-dim", type=int, default=16)
    parser.add_argument("--idm-target-action-dim", type=int, default=16)
    parser.add_argument("--idm-action-adapter", choices=["identity"], default="identity")
    parser.add_argument("--idm-custom-backend", default=None)
    parser.add_argument("--depth-backend", choices=["da3", "repeat"], default="da3")
    parser.add_argument("--da3-model-dir", default=None)
    parser.add_argument("--da3-repo-root", default=None)
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-process-res-method", default="upper_bound_resize")
    parser.add_argument("--da3-use-ray-pose", action="store_true")
    parser.add_argument("--da3-ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--wan-device", default=None)
    parser.add_argument("--ref-device", default=None)
    parser.add_argument("--depth-device", default=None)
    parser.add_argument("--idm-device", default=None)
    parser.add_argument("--hard-veto-penalty", type=float, default=50.0)
    parser.add_argument("--max-control-delta", type=float, default=0.25)
    parser.add_argument("--feasibility-weight", type=float, default=1.0)
    parser.add_argument("--action-recovery-weight", type=float, default=1.0)
    parser.add_argument("--idm-stability-weight", type=float, default=0.25)
    parser.add_argument("--transition-stability-weight", type=float, default=0.25)
    parser.add_argument("--dof-weights", default=None, help="Comma-separated per-DoF weights, e.g. 1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2")
    parser.add_argument("--override-height", type=int, default=None)
    parser.add_argument("--override-width", type=int, default=None)
    parser.add_argument("--override-n-frames", type=int, default=None)
    parser.add_argument("--override-sample-steps", type=int, default=None)
    parser.add_argument("--disable-model-compile", action="store_true")
    parser.add_argument("--disable-clip-compile", action="store_true")
    parser.add_argument("--disable-vae-compile", action="store_true")
    parser.add_argument("--disable-text-encoder-compile", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    set_random_seed(args.seed, deterministic=args.deterministic)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    wan_device = torch.device(args.wan_device if args.wan_device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ref_device = torch.device(args.ref_device if args.ref_device is not None else wan_device)
    depth_device = torch.device(args.depth_device if args.depth_device is not None else wan_device)
    idm_device = torch.device(args.idm_device if args.idm_device is not None else wan_device)

    # Ensure VAE, CLIP, and Text Encoder are assigned to the auxiliary GPU,
    # NOT the training GPU.  build_algo(cfg).to(wan_device) moves everything
    # to cuda:0 which overflows.  We explicitly relocate sub-models afterwards.
    aux_device = torch.device(args.depth_device if args.depth_device is not None else "cuda:1")

    policy_algo = build_algo(cfg).to(wan_device)

    # ── Move heavy auxiliary sub-models OFF cuda:0 to free ~29 GB ──────────
    #    DiT (14B, bf16 ≈ 28 GB) stays on cuda:0 for training.
    #    UMT5-XXL (≈ 26 GB) + CLIP (≈ 2 GB) + VAE (≈ 1 GB) → aux GPU.
    if aux_device != wan_device:
        if hasattr(policy_algo, "text_encoder") and policy_algo.text_encoder is not None:
            policy_algo.text_encoder = policy_algo.text_encoder.to(aux_device)
            print(f"[DEVICE] text_encoder → {aux_device}")
        if hasattr(policy_algo, "clip") and policy_algo.clip is not None:
            policy_algo.clip = policy_algo.clip.to(aux_device)
            print(f"[DEVICE] clip → {aux_device}")
        if hasattr(policy_algo, "vae") and policy_algo.vae is not None:
            policy_algo.vae = policy_algo.vae.to(aux_device)
            # Also move VAE-related buffers
            if hasattr(policy_algo, "vae_mean"):
                policy_algo.vae_mean = policy_algo.vae_mean.to(aux_device)
            if hasattr(policy_algo, "vae_inv_std"):
                policy_algo.vae_inv_std = policy_algo.vae_inv_std.to(aux_device)
            print(f"[DEVICE] vae → {aux_device}")

        # Override self.device so that it returns wan_device (cuda:0) instead of
        # following the first registered parameter (text_encoder, now on cuda:1).
        # PyTorch Lightning / nn.Module use this property for tensor creation.
        policy_algo.__class__ = type(
            policy_algo.__class__.__name__,
            (policy_algo.__class__,),
            {"device": property(lambda self: wan_device)},
        )

        torch.cuda.empty_cache()
        _alloc_gb = torch.cuda.memory_allocated(wan_device) / 1024**3
        print(f"[DEVICE] cuda:0 allocated after offload: {_alloc_gb:.1f} GB")

    ref_algo = None
    if args.use_reference_model:
        ref_algo = copy.deepcopy(policy_algo).to(ref_device)
        # Share frozen sub-models with policy to save GPU memory
        if aux_device != wan_device:
            ref_algo.vae = policy_algo.vae
            ref_algo.clip = policy_algo.clip
            ref_algo.text_encoder = policy_algo.text_encoder
            ref_algo.tokenizer = policy_algo.tokenizer
            if hasattr(policy_algo, "vae_mean"):
                ref_algo.vae_mean = policy_algo.vae_mean
            if hasattr(policy_algo, "vae_inv_std"):
                ref_algo.vae_inv_std = policy_algo.vae_inv_std
            # Override device property for ref_algo too
            ref_algo.__class__ = type(
                ref_algo.__class__.__name__,
                (ref_algo.__class__,),
                {"device": property(lambda self: ref_device)},
            )
        ref_algo.eval()
        for param in ref_algo.parameters():
            param.requires_grad_(False)

    optimizer = build_optimizer(policy_algo, lr=args.lr, weight_decay=args.weight_decay)

    idm_backend = build_idm_backend(
        backend_type=args.idm_backend,
        checkpoint=args.idm_checkpoint,
        model_name=args.idm_model_name,
        dinov2_name=args.idm_dinov2_name,
        left_arm_dim=args.idm_left_arm_dim,
        right_arm_dim=args.idm_right_arm_dim,
        model_output_dim=args.idm_model_output_dim,
        target_action_dim=args.idm_target_action_dim,
        action_adapter=args.idm_action_adapter,
        custom_backend_target=args.idm_custom_backend,
        device=str(idm_device),
    )
    depth_backend = build_depth_backend(
        backend_type=args.depth_backend,
        model_dir=args.da3_model_dir,
        da3_repo_root=args.da3_repo_root,
        device=str(depth_device),
        process_res=args.da3_process_res,
        process_res_method=args.da3_process_res_method,
        use_ray_pose=args.da3_use_ray_pose,
        ref_view_strategy=args.da3_ref_view_strategy,
    )
    reward_model = StepExecutabilityReward(
        action_dim=args.idm_target_action_dim,
        dof_weights=parse_optional_float_list(args.dof_weights),
        hard_veto_penalty=args.hard_veto_penalty,
        max_control_delta=args.max_control_delta,
        feasibility_weight=args.feasibility_weight,
        action_recovery_weight=args.action_recovery_weight,
        idm_stability_weight=args.idm_stability_weight,
        transition_stability_weight=args.transition_stability_weight,
    )

    trainer = StateUnrolledGRPOTrainer(
        policy_algo=policy_algo,
        ref_algo=ref_algo,
        optimizer=optimizer,
        idm_backend=idm_backend,
        depth_backend=depth_backend,
        reward_model=reward_model,
        cfg=StateUnrolledGRPOTrainerConfig(
            group_size=args.group_size,
            horizon_steps=args.horizon_steps,
            hist_len=args.hist_len,
            flow_sampling_noise_std=args.flow_sampling_noise_std,
            clip_eps=args.clip_eps,
            beta_kl=args.beta_kl,
            surrogate_sigma=args.surrogate_sigma,
            log_ratio_clip=args.log_ratio_clip,
            grad_clip_norm=args.grad_clip_norm,
            ref_update_interval=args.ref_update_interval,
            discount_gamma=args.discount_gamma,
            num_inner_epochs=args.num_inner_epochs,
        ),
    )

    condition_bank = load_condition_bank(args.condition_bank)
    if not condition_bank:
        raise ValueError(f"Condition bank is empty: {args.condition_bank}")

    history = []
    start_step = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu")
        policy_state = checkpoint.get("policy_trainable_state_dict")
        if policy_state is None:
            raise KeyError(f"Checkpoint missing policy_trainable_state_dict: {resume_path}")
        policy_algo.load_state_dict(policy_state, strict=False)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        history = checkpoint.get("history", [])
        start_step = int(checkpoint.get("step", 0))
        trainer.step = start_step
        if ref_algo is not None:
            trainer.sync_reference_policy()

    for step in range(start_step, args.steps):
        cond_batch = get_condition_batch(
            condition_bank,
            step % len(condition_bank),
            wan_device,
            n_frames=policy_algo.n_frames,
            height=policy_algo.height,
            width=policy_algo.width,
        )
        if (
            "action_seq" not in cond_batch
            and "action" in cond_batch
            and isinstance(cond_batch["action"], torch.Tensor)
            and cond_batch["action"].ndim == 2
        ):
            cond_batch["action_seq"] = cond_batch["action"].unsqueeze(1)
            cond_batch["action_seq_mask"] = torch.ones(
                cond_batch["action"].shape[0],
                1,
                device=cond_batch["action"].device,
                dtype=torch.float32,
            )
            cond_batch["valid_horizon"] = torch.ones(
                cond_batch["action"].shape[0],
                device=cond_batch["action"].device,
                dtype=torch.long,
            )
        loss_dict = trainer.train_step(cond_batch)
        record = {
            "step": step + 1,
            "loss": float(loss_dict.loss.detach().cpu()),
            "policy_loss": float(loss_dict.policy_loss.detach().cpu()),
            "kl_loss": float(loss_dict.kl_loss.detach().cpu()),
            "mean_ratio": float(loss_dict.mean_ratio.detach().cpu()),
            "mean_advantage": float(loss_dict.mean_advantage.detach().cpu()),
            "group_reward_mean": float(loss_dict.group_reward_mean.detach().cpu()),
            "group_reward_std": float(loss_dict.group_reward_std.detach().cpu()),
            "grad_norm": float(loss_dict.grad_norm.detach().cpu()) if loss_dict.grad_norm is not None else 0.0,
        }
        history.append(record)

        if (step + 1) % args.log_interval == 0:
            print(json.dumps(record))
        if (step + 1) % args.save_interval == 0:
            checkpoint = {
                "step": step + 1,
                "policy_trainable_state_dict": extract_trainable_state_dict(policy_algo),
                "optimizer_state_dict": _move_state_to_cpu(optimizer.state_dict()),
                "history": history,
            }
            checkpoint_path = save_dir / f"state_unrolled_grpo_step_{step + 1}.pt"
            try:
                torch.save(checkpoint, checkpoint_path)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "warning": "checkpoint_save_failed",
                            "path": str(checkpoint_path),
                            "error": str(exc),
                        }
                    )
                )

    with (save_dir / "state_unrolled_grpo_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
