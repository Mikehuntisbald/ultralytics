# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .muon import Muon, MuSGD


def get_prodigy_optimizer():
    """Return the optional Prodigy optimizer class, raising a clear install hint when unavailable."""
    try:
        from prodigyopt import Prodigy
    except ImportError as e:
        raise ImportError(
            "optimizer=Prodigy requires the optional 'prodigyopt' package. "
            "Install it with `pip install prodigyopt`."
        ) from e
    return Prodigy


__all__ = ["MuSGD", "Muon", "get_prodigy_optimizer"]
