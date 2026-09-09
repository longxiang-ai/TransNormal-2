#!/usr/bin/env python
"""TransNormal-2 inference on an image or a folder of images.

Example:
    python inference.py \
        --input path/to/transparent_image.jpg \
        --output_dir outputs \
        --weights Longxiang-ai/TransNormal-2 \
        --domain transparent
"""

import argparse
import os
import time
from pathlib import Path

import torch

from transnormal2 import TransNormal2Pipeline, load_image, save_normal_map


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="TransNormal-2 inference on an image or a folder")
    p.add_argument("--input", "--image", "--input_dir", dest="input", required=True,
                   help="Input RGB image or directory (searched recursively)")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--weights", type=str, default="Longxiang-ai/TransNormal-2",
                   help="Local weights dir or HF repo id (e.g. Longxiang-ai/TransNormal-2)")
    p.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.2-klein-base-9B",
                   help="FLUX.2 [klein] base model (HF id or local path)")
    p.add_argument("--domain", choices=["opaque", "transparent"], default="opaque",
                   help="Scene domain for GRM anchoring; use 'transparent' for glass/liquid-dominant scenes")
    p.add_argument("--process_res", type=int, default=None,
                   help="Optional max processing edge (default: native resolution)")
    p.add_argument("--no_grm", action="store_true", help="Skip GRM refinement (raw prediction)")
    p.add_argument("--save_npy", action="store_true", help="Also save the raw [0,1] prediction as .npy")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16 recommended; fp16 is intentionally not offered (NaN risk)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Enable model CPU offload to reduce peak GPU memory")
    return p.parse_args()


def resolve_weights(weights: str) -> str:
    if os.path.isdir(weights):
        return weights
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=weights, allow_patterns=["*.safetensors", "config.json"],
        token=False if weights == "Longxiang-ai/TransNormal-2" else None,
    )


def collect_images(input_path: str, output_dir: str):
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image format: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if source == output:
        raise ValueError("Use separate input and output directories.")
    images = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and not p.is_relative_to(output)
    )
    if not images:
        raise ValueError("No supported input images were found.")
    destinations = [p.relative_to(source).with_suffix("") for p in images]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Images in the same folder must have distinct filename stems.")
    return images, source


def main():
    args = parse_args()
    images, input_root = collect_images(args.input, args.output_dir)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    weights_dir = resolve_weights(args.weights)
    pipe = TransNormal2Pipeline.from_pretrained_transnormal2(
        base_model=args.base_model,
        weights_dir=weights_dir,
        torch_dtype=dtype,
        device=None if args.cpu_offload else args.device,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=args.device)
        pipe.local_continuity_module.to(args.device)
        pipe.grm.to(args.device)

    for i, img_path in enumerate(images, 1):
        image = load_image(str(img_path))
        start = time.time()
        normal = pipe(
            image,
            domain_is_transparent=(args.domain == "transparent"),
            process_res=args.process_res,
            apply_grm=not args.no_grm,
            output_type="pt",
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        folder = Path(args.output_dir) / img_path.relative_to(input_root).parent
        folder.mkdir(parents=True, exist_ok=True)
        out_png = str(folder / f"{img_path.stem}_normal.png")
        out_npy = str(folder / f"{img_path.stem}_normal.npy") if args.save_npy else None
        save_normal_map(normal, out_png, save_npy=out_npy)
        print(f"[{i}/{len(images)}] {img_path.name} -> {out_png} ({time.time() - start:.2f}s)")

    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
