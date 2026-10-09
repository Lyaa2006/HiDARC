#!/usr/bin/env python3
"""Fit the fixed InternViT PCA basis used by HiDESC spectral routing.

Example:
  python scripts/MCITlib/fit_spectral_pca.py \
    --vision_tower /path/to/InternViT-6B-224px \
    --data_json /path/to/calibration.json \
    --image_folder /path/to/images \
    --output_path /path/to/internvl_spectral_pca_3200_to_512.pt
"""

import argparse
import json
import os
import sys
from typing import List

import torch
from PIL import Image, ImageFile
from transformers import CLIPImageProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.mm_utils import expand2square
from llava.model.multimodal_encoder.clip_encoder import is_intern_vit_6b_model
from llava.model.multimodal_encoder.intern_vit_6b.configuration_intern_vit import InternVisionConfig
from llava.model.multimodal_encoder.intern_vit_6b.modeling_intern_vit import InternVisionModel
from llava.model.spectral_pca import FixedPCARouteProjector

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _image_paths(data_json: str, image_folder: str, max_images: int) -> List[str]:
    with open(data_json, encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise TypeError("data_json must contain a JSON list")
    paths = []
    for sample in samples:
        if not isinstance(sample, dict) or not sample.get("image"):
            continue
        raw = sample["image"]
        path = raw if os.path.isabs(raw) else os.path.join(image_folder or "", raw)
        if os.path.isfile(path):
            paths.append(os.path.abspath(path))
        else:
            raise FileNotFoundError(f"Image referenced by calibration data is missing: {path}")
        if max_images > 0 and len(paths) >= max_images:
            break
    if not paths:
        raise ValueError("No image samples were found in data_json")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision_tower", required=True)
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--max_patches", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--vision_select_layer", type=int, default=-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not is_intern_vit_6b_model(args.vision_tower):
        raise ValueError("This PCA fitter is for InternViT-6B; CLIP already has a projected route width")
    config = InternVisionConfig.from_pretrained(args.vision_tower)
    image_size = int(config.image_size)
    processor = CLIPImageProcessor(
        crop_size=image_size,
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size=image_size,
    )
    model_dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    model = InternVisionModel.from_pretrained(args.vision_tower).to(
        device=args.device, dtype=model_dtype
    )
    model.eval()
    paths = _image_paths(args.data_json, args.image_folder, args.max_images)
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), max(1, args.batch_size)):
            batch_paths = paths[start : start + max(1, args.batch_size)]
            batch = []
            for path in batch_paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    batch.append(processor.preprocess(image, return_tensors="pt")["pixel_values"][0])
            pixel_values = torch.stack(batch).to(device=args.device, dtype=model.dtype)
            outputs = model(pixel_values, output_hidden_states=True)
            selected = outputs.hidden_states[args.vision_select_layer][:, 1:].float().cpu()
            if args.max_patches > 0:
                selected = selected[:, : args.max_patches]
            chunk = selected.reshape(-1, selected.shape[-1])
            if args.max_samples > 0:
                remaining = args.max_samples - sum(x.shape[0] for x in features)
                if remaining <= 0:
                    break
                chunk = chunk[:remaining]
            features.append(chunk)
            print(f"processed {min(start + len(batch_paths), len(paths))}/{len(paths)} images", flush=True)
            if args.max_samples > 0 and sum(x.shape[0] for x in features) >= args.max_samples:
                break
    samples = torch.cat(features, dim=0)
    projector = FixedPCARouteProjector.fit(
        samples,
        args.output_dim,
        metadata={
            "vision_tower": os.path.abspath(args.vision_tower),
            "vision_select_layer": args.vision_select_layer,
            "vision_select_feature": "patch",
            "input_dim": int(samples.shape[-1]),
            "output_dim": args.output_dim,
            "image_size": image_size,
            "patch_count": int(selected.shape[1]),
            "dtype": "float32",
            "fit_dataset": os.path.abspath(args.data_json),
            "sample_count": int(samples.shape[0]),
        },
    )
    projector.save(args.output_path)
    print(f"saved PCA: {args.output_path} ({projector.input_dim} -> {projector.output_dim}, {samples.shape[0]} patches)")


if __name__ == "__main__":
    main()
