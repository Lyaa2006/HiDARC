# InternVL HiDARC configuration

The InternVL HiDARC implementation lives in `InternVL/HiDARC` and uses the
same description-aligned losses and role-aware progressive routing as the
LLaVA implementation.  InternVL-specific stage bands are `B1=[12,14]` and
`B2=[27,29]`; the decoder has 32 layers.

Install the pinned Python dependencies once in the dedicated environment:

```bash
python -m pip install -r InternVL/HiDARC/requirements.txt
```

The training launchers load the shared environment preflight automatically;
they do not install packages during a run.

Use the mainline launchers under `InternVL/HiDARC/scripts/MCITlib/Train/`.
They generate run-local task configurations so that output paths and previous
task dependencies are explicit and reproducible.  No ablation configuration
is used by these launchers.

Before either training launcher, set `CACHE_ROOT` to the selected
description-cache root.  The launcher validates/generates the current task's
description cache one task at a time.  It also creates and reuses a fixed
InternVL spectral PCA file at the root when `SPECTRAL_PCA_PATH` is not set;
the first task's image data is used as the calibration set, so no additional
configuration field is required.

Evaluation entry points are `Eval_UCIT/eval_ucit.sh` and
`Eval_MLLM_DCL/eval_dcl.sh`. They require explicit `CHECKPOINT_ROOT` and
`RESULT_ROOT` environment variables. The InternVL stage-1 band schedule is
fixed in the HiDARC model code, rather than injected from an evaluation JSON.
They do not provide a checkpoint default and do not reference the ablation
tree.

## Fixed spectral PCA

InternViT exposes 3200-dimensional patch features, while HiDARC uses a
separate image spectral route. The route can apply one offline-fitted PCA basis
to reduce this channel width (the recommended first run uses 512):

```text
InternViT patch features [B, N, 3200]
    -> fixed PCA [B, N, 512]
    -> 8 spectral bands
    -> image route descriptor [B, 4096]
```

Fit the basis once on a representative calibration image set, then pass the
same absolute file to every UCIT/DCL task:

```bash
python InternVL/HiDARC/scripts/MCITlib/fit_spectral_pca.py \
  --vision_tower /your_model_path/InternVL/InternViT-6B-224px \
  --data_json /path/to/calibration.json \
  --image_folder /path/to/images \
  --output_path /path/to/internvl_spectral_pca_3200_to_512.pt \
  --output_dim 512

SPECTRAL_PCA_PATH=/path/to/internvl_spectral_pca_3200_to_512.pt \
  bash InternVL/HiDARC/scripts/MCITlib/Train/train_UCIT.sh
```

Set `spectral_route_channel_dim` to the basis output width in each generated
task configuration. The PCA only changes `projected_patch_features` used for
spectral routing; language-side `selected_patch_features` remain 3200
dimensional and continue to use the existing InternVL projector. A missing or
dimension-mismatched PCA file fails fast so training and evaluation cannot
silently produce incompatible prototype spaces.
