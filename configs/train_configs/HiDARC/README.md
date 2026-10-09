# HiDARC training configuration

The four `final.json` profiles below record run resources and optimization
settings. They follow the repository convention used by the other methods:
`rank` is the LoRA rank and `expert_num` is the number of task experts. The
launchers derive `lora_alpha = 2 * rank`.

| Profile | Reference run | Model | Protocol |
| --- | --- | --- | --- |
| `LLaVA/UCIT/train/final.json` | 2026-08-27 `hidesc_new_strategy_full_20260827_103909` | LLaVA | UCIT, 6 tasks |
| `LLaVA/MLLM-DCL/train/final.json` | 2026-08-27 `hidesc_new_strategy_full_20260827_103909` | LLaVA | DCL, 5 tasks |
| `InternVL/UCIT/train/final.json` | 2026-09-18 `ucit_gpu7_rolebank829_last_eval_20260918` | InternVL | UCIT, 6 tasks |
| `InternVL/MLLM-DCL/train/final.json` | 2026-09-25 native 3-role matrix | InternVL | DCL, 5 tasks |

Each `<model>/<protocol>/collaboration.json` is the canonical routing and
role-assignment profile for that experiment family.  The task-specific
`task*.json` files retain only task/resource differences such as `cur_task`,
checkpoint, and description-cache paths.  The train launchers materialize the
task config with its sibling collaboration profile; eval launchers pass that
same profile through `--routing-config-path`.

The final FFT anchor extraction and fixed stage-band policy are implemented in
`LLaVA/HiDARC/llava/model/hidarc_final.py` and
`InternVL/HiDARC/llava/model/hidarc_final.py`; those fixed values are not in
JSON. The `final.json` profiles retain only optimization/resource parameters.
The anchor is the online
sample-weighted mean of the same per-example descriptor used offline, so it is
independent of batch partition. A later task rejects a prior checkpoint unless
its persisted anchor/role profile matches exactly.
