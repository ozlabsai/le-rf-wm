"""Render numpy spectrograms as base64 PNG strings."""

import io
import base64

import numpy as np
from PIL import Image


# Viridis colormap LUT (256 entries, RGB)
# Generated from matplotlib: plt.cm.viridis(np.linspace(0, 1, 256))[:, :3]
def _build_viridis_lut():
    try:
        import matplotlib.pyplot as plt
        cmap = plt.cm.viridis
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple yellow-blue gradient
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            lut[i] = [int(68 * (1-t) + 253 * t),
                       int(1 * (1-t) + 231 * t),
                       int(84 * (1-t) + 37 * t)]
    return lut


VIRIDIS_LUT = _build_viridis_lut()


def render_frame(array, vmin=0.0, vmax=1.0):
    """Convert a 2D float array to a base64-encoded PNG string.

    Args:
        array: (H, W) numpy array, expected [0, 1] range
        vmin, vmax: clipping bounds for normalization

    Returns:
        base64 string (no data: prefix)
    """
    arr = np.asarray(array, dtype=np.float32)
    # Flip vertically so low frequencies are at bottom (origin="lower")
    arr = arr[::-1]
    # Normalize to [0, 255]
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    indices = (arr * 255).astype(np.uint8)
    # Apply colormap
    rgb = VIRIDIS_LUT[indices]  # (H, W, 3)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
