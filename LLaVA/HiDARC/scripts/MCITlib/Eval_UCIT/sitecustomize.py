"""Disable optional bitsandbytes discovery for unquantized inference smoke."""

import importlib.util

_original_find_spec = importlib.util.find_spec


def _find_spec_without_bitsandbytes(name, *args, **kwargs):
    if name == "bitsandbytes" or name.startswith("bitsandbytes."):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_bitsandbytes
