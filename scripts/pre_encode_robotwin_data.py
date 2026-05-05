"""Pre-compute VAE latents & T5 embeddings for all MP4 episodes offline.

This eliminates online VAE+T5 encoding during training (saves ~8-10s/iter).

Usage:
  python scripts/pre_encode_robotwin_data.py \
    --data-dir /data1/zmh/robotwin_sft_vanilla \
    --wan-ckpt /path/to/Wan2.1-I2V-14B-480P \
    --device cuda:0 \
    --batch-size 4   # VAE chunk size for long videos
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.wan.modules.vae import video_vae_factory
from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer


def load_vae(ckpt_path: str, z_dim: int = 16, dtype=torch.bfloat16):
    model = (
        video_vae_factory(pretrained_path=ckpt_path, z_dim=z_dim)
        .eval()
        .requires_grad_(False)
        .to(dtype=dtype)
    )
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
        1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
        0.2503, -0.2921
    ], dtype=dtype)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype)
    inv_std = (1.0 / std).to(dtype=dtype)
    return model, mean, inv_std


def load_t5(ckpt_path: str, tokenizer_name: str, dtype=torch.bfloat16):
    encoder = (
        umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=dtype, device=torch.device("cpu"))
        .eval()
        .requires_grad_(False)
    )
    if ckpt_path:
        encoder.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
    tokenizer = HuggingfaceTokenizer(name=tokenizer_name, seq_len=512, clean="whitespace")
    return encoder, tokenizer


def caption_hash(caption: str) -> str:
    return hashlib.md5(caption.encode()).hexdigest()[:12]


@torch.no_grad()
def encode_text(encoder, tokenizer, texts: list[str], device: torch.device):
    # Use tokenizer on ALL texts at once — matches training code exactly
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)

    encoder = encoder.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = encoder(ids, mask)
    encoder = encoder.cpu()
    return [u[:v] for u, v in zip(context, seq_lens)], seq_lens


@torch.no_grad()
def encode_video_full(vae_model, mean, inv_std, frames: torch.Tensor, device: torch.device):
    """Encode all frames of a video through VAE.

    Args:
        frames: [T, C, H, W] in range [0, 1]
    Returns:
        latent: [C_lat, T_lat, H_lat, W_lat]
    """
    # Normalize to [-1, 1], rearrange to [C, T, H, W], and cast to VAE dtype
    frames = (frames * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, C, T, H, W]
    latent = vae_model.encode(frames, [mean.to(device), inv_std.to(device)]).float().squeeze(0).cpu()
    return latent


def frame_to_latent_idx(frame_idx: int) -> int:
    """Map frame index to VAE latent temporal index (stride=4)."""
    if frame_idx <= 0:
        return 0
    return 1 + (frame_idx - 1) // 4


def main():
    parser = argparse.ArgumentParser(description="Pre-encode VAE latents and T5 embeddings")
    parser.add_argument("--data-dir", required=True, help="Dataset directory with videos/ and metadata.csv")
    parser.add_argument("--wan-ckpt", required=True, help="Path to Wan2.1-I2V-14B-480P checkpoint directory")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--batch-size", type=int, default=1, help="Videos to encode at once (keep low for long videos)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video to encode (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip videos that already have latent files")
    parser.add_argument("--workers", type=int, default=0, help="(reserved)")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "metadata.csv"
    latent_dir = data_dir / "latents"
    prompt_dir = data_dir / "prompts"
    latent_dir.mkdir(exist_ok=True)
    prompt_dir.mkdir(exist_ok=True)

    # Load metadata
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Loaded {len(records)} records from {csv_path}")

    # Load models
    print("Loading VAE...")
    vae, vae_mean, vae_inv_std = load_vae(
        f"{args.wan_ckpt}/Wan2.1_VAE.pth", z_dim=16
    )
    vae = vae.to(device)
    vae_scale = [vae_mean.to(device), vae_inv_std.to(device)]

    print("Loading T5 encoder...")
    t5, tokenizer = load_t5(
        ckpt_path=f"{args.wan_ckpt}/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_name=f"{args.wan_ckpt}/google/umt5-xxl",
    )

    # Phase 1: Pre-compute T5 embeddings for ALL unique captions
    print("\n=== Phase 1: T5 text embeddings ===")
    unique_captions = list(set(r["caption"] for r in records))
    print(f"Unique captions: {len(unique_captions)}")

    prompt_cache: Dict[str, Path] = {}  # caption → prompt_embed_path
    t5_dtype = next(t5.parameters()).dtype

    for caption in tqdm(unique_captions, desc="T5 encoding"):
        ch = caption_hash(caption)
        pt_path = prompt_dir / f"{ch}.pt"
        if args.resume and pt_path.exists():
            prompt_cache[caption] = pt_path
            continue

        embeds, seq_lens = encode_text(t5, tokenizer, [caption], device)
        torch.save(embeds[0].cpu(), pt_path)
        prompt_cache[caption] = pt_path

    # Phase 2: Pre-compute VAE latents for each video
    print("\n=== Phase 2: VAE latents ===")
    import decord
    decord.bridge.set_bridge("torch")

    skipped = 0
    failed = 0
    encoded = 0

    for i, record in enumerate(tqdm(records, desc="VAE encoding")):
        video_rel = record["video_path"]
        video_name = Path(video_rel).stem
        latent_path = latent_dir / f"{video_name}_latent.pt"

        if args.resume and latent_path.exists():
            skipped += 1
            continue

        video_path = data_dir / video_rel
        if not video_path.exists():
            print(f"  [WARN] Missing video: {video_path}", file=sys.stderr)
            failed += 1
            continue

        try:
            vr = decord.VideoReader(str(video_path))
            all_frames = vr.get_batch(list(range(len(vr))))  # [T, H, W, C]
            all_frames = all_frames.float().permute(0, 3, 1, 2) / 255.0  # [T, C, H, W]

            if args.max_frames > 0 and all_frames.shape[0] > args.max_frames:
                all_frames = all_frames[:args.max_frames]

            latent = encode_video_full(vae, vae_mean, vae_inv_std, all_frames, device)
            torch.save(latent, latent_path)
            encoded += 1
        except Exception as e:
            print(f"  [WARN] {video_name}: {e}", file=sys.stderr)
            failed += 1

    # Phase 3: Update CSV
    print("\n=== Phase 3: Updating metadata ===")
    new_fieldnames = list(records[0].keys())
    if "video_latent_path" not in new_fieldnames:
        new_fieldnames.append("video_latent_path")
    if "prompt_embed_path" not in new_fieldnames:
        new_fieldnames.append("prompt_embed_path")
    if "prompt_embed_len" not in new_fieldnames:
        new_fieldnames.append("prompt_embed_len")

    updated = 0
    for record in records:
        video_name = Path(record["video_path"]).stem
        latent_path = latent_dir / f"{video_name}_latent.pt"
        if latent_path.exists():
            record["video_latent_path"] = str(latent_path.relative_to(data_dir))
        else:
            record["video_latent_path"] = ""

        caption = record["caption"]
        if caption in prompt_cache:
            pt_path = prompt_cache[caption]
            record["prompt_embed_path"] = str(pt_path.relative_to(data_dir))
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            record["prompt_embed_len"] = str(data.shape[0])
            updated += 1

    # Write updated CSV
    csv_backup = csv_path.with_suffix(".csv.bak")
    csv_path.rename(csv_backup)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone! VAE: {encoded} encoded, {skipped} skipped, {failed} failed")
    print(f"T5: {len(unique_captions)} captions encoded")
    print(f"Updated {updated}/{len(records)} records with prompt_embed_path")
    print(f"Original CSV backed up to {csv_backup}")


if __name__ == "__main__":
    main()
