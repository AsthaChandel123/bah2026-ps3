"""Partial-convolution U-Net interface for nonlinear cloud-gap inpainting.

This is the Stage-2 nonlinear refinement layer (synthesis blueprint method 7):
a partial-convolution U-Net (Liu et al., 2018) that reconstructs cloud / orbit
gaps in satellite column and AOD fields while preserving plume structure, run
*downstream of* the DINEOF first pass. Partial convolutions renormalise each
conv by the valid-pixel fraction in its receptive field and progressively update
the mask, so the network never "sees" the missing region as zeros.

Heavy-dependency policy (DEV_CONTRACT §3): :mod:`torch` is imported lazily inside
the methods that need it. The package therefore imports under the light set
alone. When torch is unavailable — or when no trained weights are supplied —
:func:`inpaint` transparently falls back to :func:`aqi_india.fusion.dineof.dineof`
so the demo always produces a filled field. The class below defines the model
architecture so a training run (heavy ``deep`` extra) can instantiate it, but the
public entrypoint is the function :func:`inpaint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger("fusion.inpaint_unet")


@dataclass
class InpaintResult:
    """Result of an inpainting reconstruction.

    Attributes:
        filled: Gap-filled array, same shape as the input.
        method: ``"pconv_unet"`` if the neural model ran, else ``"dineof"``.
        uncertainty: Per-cell uncertainty (from the DINEOF fallback when used,
            otherwise a distance-to-valid heuristic).
    """

    filled: np.ndarray
    method: str
    uncertainty: np.ndarray


def build_pconv_unet(
    in_channels: int = 1,
    *,
    base_filters: int = 32,
    depth: int = 4,
):
    """Construct a partial-convolution U-Net (requires :mod:`torch`).

    The model is returned only for the heavy training path; the demo never calls
    this. Importing torch is deferred to here so the module stays light.

    Args:
        in_channels: Number of input channels (e.g. 1 per species cube slice).
        base_filters: Channel width of the first encoder block.
        depth: Number of down/up-sampling stages.

    Returns:
        A ``torch.nn.Module`` implementing partial-conv encode/decode with mask
        propagation.

    Raises:
        ImportError: If :mod:`torch` is not installed.
    """
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - heavy path
        raise ImportError(
            "build_pconv_unet requires torch (install the 'deep' extra)"
        ) from exc

    class PartialConv2d(nn.Module):
        """2-D partial convolution with automatic mask update."""

        def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
            super().__init__()
            pad = k // 2
            self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=True)
            self.register_buffer("weight_maskUpdater", torch.ones(1, 1, k, k))
            self.stride, self.pad, self.k = stride, pad, k
            self.slide_winsize = float(k * k)

        def forward(self, x, mask):  # noqa: ANN001
            with torch.no_grad():
                update_mask = nn.functional.conv2d(
                    mask, self.weight_maskUpdater, stride=self.stride, padding=self.pad
                )
                ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)
                ratio = ratio * update_mask
            out = self.conv(x * mask) * ratio
            return out, update_mask

    class PConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int, down: bool = True):
            super().__init__()
            self.pconv = PartialConv2d(cin, cout, 3, stride=2 if down else 1)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.ReLU() if down else nn.LeakyReLU(0.2)

        def forward(self, x, mask):  # noqa: ANN001
            x, mask = self.pconv(x, mask)
            return self.act(self.bn(x)), mask

    class PConvUNet(nn.Module):
        """Partial-convolution U-Net (encoder/decoder with skip connections)."""

        def __init__(self, cin: int, base: int, depth: int):
            super().__init__()
            self.depth = depth
            self.enc = nn.ModuleList()
            ch = cin
            for d in range(depth):
                out = base * (2**d)
                self.enc.append(PConvBlock(ch, out, down=True))
                ch = out
            self.dec = nn.ModuleList()
            for d in reversed(range(depth)):
                skip = cin if d == 0 else base * (2 ** (d - 1))
                out = skip
                self.dec.append(PConvBlock(ch + skip, out, down=False))
                ch = out
            self.final = nn.Conv2d(ch, cin, 1)

        def forward(self, x, mask):  # noqa: ANN001
            feats = [(x, mask)]
            for blk in self.enc:
                x, mask = blk(x, mask)
                feats.append((x, mask))
            for blk in self.dec:
                skip_x, skip_m = feats.pop(-2) if len(feats) > 1 else feats[-1]
                x = nn.functional.interpolate(x, size=skip_x.shape[-2:], mode="nearest")
                mask = nn.functional.interpolate(
                    mask, size=skip_m.shape[-2:], mode="nearest"
                )
                x = torch.cat([x, skip_x], dim=1)
                mask = torch.cat([mask, skip_m], dim=1)
                x, mask = blk(x, mask)
            return self.final(x)

    return PConvUNet(in_channels, base_filters, depth)


def _distance_uncertainty(missing: np.ndarray) -> np.ndarray:
    """Heuristic uncertainty: normalised distance from each cell to valid data.

    Used when the neural model runs (which does not emit a native variance).
    """
    from scipy.ndimage import distance_transform_edt

    if missing.ndim == 2:
        dist = distance_transform_edt(missing)
        return (dist / (dist.max() + 1e-9)).astype("float32")
    out = np.zeros_like(missing, dtype="float32")
    for i in range(missing.shape[0]):
        d = distance_transform_edt(missing[i])
        out[i] = d / (d.max() + 1e-9)
    return out


def inpaint(
    data: np.ndarray,
    *,
    weights_path: str | None = None,
    device: str = "cpu",
    dineof_k_max: int = 20,
    seed: int = 42,
) -> InpaintResult:
    """Inpaint cloud / orbit gaps, preferring the U-Net, falling back to DINEOF.

    This is the public entrypoint. If :mod:`torch` and trained ``weights_path``
    are both available, the partial-conv U-Net reconstructs the gaps; otherwise
    the function falls back to :func:`aqi_india.fusion.dineof.dineof` (label-free,
    light deps) so it always succeeds in the demo environment.

    Args:
        data: Array with ``NaN`` gaps, shape ``(time, lat, lon)`` or ``(lat, lon)``.
        weights_path: Path to trained model weights. If ``None``, the DINEOF
            fallback is used (the demo path).
        device: Torch device string for the neural path (e.g. ``"cpu"``, ``"cuda"``).
        dineof_k_max: Max EOF modes passed to the DINEOF fallback.
        seed: RNG seed forwarded to the fallback.

    Returns:
        An :class:`InpaintResult`.
    """
    arr = np.asarray(data, dtype="float64")
    missing = ~np.isfinite(arr)

    if weights_path is not None:
        try:
            filled = _inpaint_torch(arr, missing, weights_path, device)
            return InpaintResult(
                filled=filled.astype("float32"),
                method="pconv_unet",
                uncertainty=_distance_uncertainty(missing),
            )
        except Exception as exc:
            logger.warning(
                "Partial-conv U-Net inpaint failed (%s); falling back to DINEOF",
                type(exc).__name__,
            )

    from .dineof import dineof

    # DINEOF expects a leading sample axis; add one for a single 2-D frame.
    if arr.ndim == 2:
        res = dineof(arr[None], k_max=dineof_k_max, seed=seed)
        return InpaintResult(
            filled=res.filled[0], method="dineof", uncertainty=res.uncertainty[0]
        )
    res = dineof(arr, k_max=dineof_k_max, seed=seed)
    return InpaintResult(filled=res.filled, method="dineof", uncertainty=res.uncertainty)


def _inpaint_torch(
    arr: np.ndarray,
    missing: np.ndarray,
    weights_path: str,
    device: str,
) -> np.ndarray:
    """Run the partial-conv U-Net forward pass (heavy path; lazy torch).

    Args:
        arr: Input array with NaN gaps (``(time, lat, lon)`` or ``(lat, lon)``).
        missing: Boolean mask, True where missing.
        weights_path: Path to a ``state_dict`` checkpoint.
        device: Torch device string.

    Returns:
        The gap-filled array (present pixels preserved exactly).
    """
    import torch

    frames = arr[None] if arr.ndim == 2 else arr
    mframe = missing[None] if missing.ndim == 2 else missing

    model = build_pconv_unet(in_channels=1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.to(device).eval()

    filled = np.array(frames, dtype="float64")
    fill_val = float(np.nanmean(frames)) if np.isfinite(frames).any() else 0.0
    with torch.no_grad():
        for i in range(frames.shape[0]):
            x = np.where(mframe[i], fill_val, frames[i])
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)[None, None]
            mt = torch.as_tensor(
                (~mframe[i]).astype("float32"), device=device
            )[None, None]
            pred = model(xt, mt).cpu().numpy()[0, 0]
            filled[i] = np.where(mframe[i], pred, frames[i])

    return filled[0] if arr.ndim == 2 else filled


__all__ = ["InpaintResult", "build_pconv_unet", "inpaint"]
