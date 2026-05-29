"""Runtime compatibility shims for optional dependency combinations."""

from __future__ import annotations


def patch_optional_dependencies() -> None:
    """Disable optional Transformers torchvision paths when torchvision is incomplete.

    Some cluster environments expose a partial or mismatched torchvision package.
    Transformers treats that as available and imports video/image helpers eagerly
    while loading Trainer, which can fail before this project reaches any HAMU
    code. HAMU does not require torchvision-backed fast processors, so marking it
    unavailable keeps the public CLI usable in those environments.
    """

    try:
        import torchvision  # noqa: F401
        from torchvision import transforms

        torchvision_ok = hasattr(torchvision, "io") and hasattr(transforms, "InterpolationMode")
    except Exception:
        torchvision_ok = False

    if torchvision_ok:
        return

    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    if hasattr(import_utils.is_torchvision_available, "cache_clear"):
        import_utils.is_torchvision_available.cache_clear()
    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False
