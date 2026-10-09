"""Fixed HiDARC anchor and layer-band policy for InternVL.

Role assignment and activation values are experiment-specific and remain in
each final training profile.
"""

from copy import deepcopy

CANONICAL_ROUTING_STRATEGY = "prototype_only_role_constrained_late"
FIXED_HIDARC_CONFIG = {
    "spectral_cutoff": 0.33,
    "spectral_low_bins": 4,
    "spectral_high_bins": 4,
    "spectral_image_ema_decay": 0.8,
    "text_activation_ema_decay": 0.8,
    "text_activation_highpass_exponent": 0.75,
    "text_activation_magnitude_weight": 0.0,
    "text_activation_real_weight": 0.5,
    "text_activation_imag_weight": 0.5,
    "text_activation_use_fftshift": True,
    "routing_strategy": CANONICAL_ROUTING_STRATEGY,
    "use_stage1_band_schedule_eval": True,
}
FIXED_HIDARC_KEYS = frozenset(FIXED_HIDARC_CONFIG)

def get_fixed_hidarc_config():
    return deepcopy(FIXED_HIDARC_CONFIG)
