"""Lazy singleton for the imagination pipeline."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mae"))

_pipeline = None
_init_error = None


def get_pipeline():
    """Return the RFWorldModelImagination instance, or None with error message."""
    global _pipeline, _init_error

    if _pipeline is not None:
        return _pipeline
    if _init_error is not None:
        return None

    try:
        from imagine import RFWorldModelImagination

        data_dir = Path(os.environ.get("DEMO_DATA_DIR", str(PROJECT_ROOT / "demo" / "data")))
        mae_dir = PROJECT_ROOT / "mae"

        wm_ckpt = os.environ.get("WM_CKPT", str(data_dir / "lewm_rf_epoch_100_object.ckpt"))
        bridge_ckpt = os.environ.get("BRIDGE_CKPT", str(mae_dir / "bridge_best.ckpt"))
        wm_norm = os.environ.get("WM_NORM_STATS", str(data_dir / "norm_stats.json"))
        mae_norm = os.environ.get("MAE_NORM_STATS", str(mae_dir / "cache" / "norm_stats.json"))

        # Check files exist
        for label, path in [("WM checkpoint", wm_ckpt),
                            ("Bridge checkpoint", bridge_ckpt), ("WM norm stats", wm_norm),
                            ("MAE norm stats", mae_norm)]:
            if not Path(path).exists():
                _init_error = f"Missing {label}: {path}"
                print(f"Imagination pipeline unavailable: {_init_error}")
                return None

        _pipeline = RFWorldModelImagination(
            wm_checkpoint=wm_ckpt,
            bridge_checkpoint=bridge_ckpt,
            norm_stats_path=wm_norm,
            mae_norm_stats_path=mae_norm,
        )
        print("Imagination pipeline loaded.")
        return _pipeline

    except Exception as e:
        _init_error = str(e)
        print(f"Imagination pipeline failed to load: {_init_error}")
        return None


def get_init_error():
    return _init_error
