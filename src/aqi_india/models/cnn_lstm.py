"""CNN-LSTM hybrid — the PS-mandated workhorse Objective-1 model (PyTorch).

The CNN-LSTM hybrid is the PS3-named, safe primary baseline for daily surface
estimation (literature R2 ~0.91, RMSE ~8.2 on clean data): a 2-D CNN encodes the
spatial neighbourhood of satellite/met/physics channels at each time step, and an
LSTM consumes the resulting per-step feature sequence to capture pollutant
memory — nighttime BLH collapse, multi-day accumulation lags — before a head maps
to the surface field.

Two complementary output modes share one encoder:

* **gridded** (``mode="grid"``): input ``[B, T, C, H, W]`` -> per-cell output
  ``[B, P, H, W]``. A fully-convolutional CNN keeps the spatial dims; the LSTM is
  applied per cell (channels-as-features along ``T``) so the whole grid is
  predicted coherently — the analogue of the SA-ConvLSTM path on a lighter
  budget.
* **patch/point** (``mode="point"``): input ``[B, T, C, H, W]`` patches centred
  on CPCB stations -> scalar-per-pollutant output ``[B, P]``. The CNN pools each
  patch to a vector, the LSTM runs over ``T``, and an MLP head regresses the
  station value — the classical CNN-LSTM column->surface mapping.

**Importability:** ``torch`` is imported lazily inside the factory so this module
imports under the light dependency set. The demo uses LightGBM; this model
satisfies the PS3 CNN/LSTM mandate and runs when the ``deep`` extra is installed.

Shape convention: ``B`` batch, ``T`` time, ``C`` input channels, ``P`` output
pollutants, ``H``/``W`` grid/patch rows/cols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

_log = get_logger("models.cnn_lstm")


def _torch_modules() -> tuple[Any, Any, Any]:
    """Lazily import and return ``(torch, nn, F)``.

    Raises:
        ImportError: With install guidance if torch is unavailable.
    """
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        raise ImportError(
            "PyTorch is required for CNN-LSTM. Install the deep extra: "
            "`pip install -e .[deep]`."
        ) from exc
    return torch, nn, F


def build_cnn_lstm(
    in_channels: int,
    out_channels: int,
    *,
    cnn_channels: "tuple[int, ...]" = (32, 64),
    hidden_size: int = 128,
    kernel_size: int = 3,
    num_lstm_layers: int = 1,
    mode: str = "grid",
) -> "nn.Module":
    """Factory constructing a ready-to-train CNN-LSTM hybrid module.

    Defined as a factory so the ``nn.Module`` subclasses are created only when
    torch is importable, keeping the module import-light.

    Args:
        in_channels: Number of input channels ``C``.
        out_channels: Number of output pollutants ``P``.
        cnn_channels: Output channels of each CNN encoder block.
        hidden_size: LSTM hidden size.
        kernel_size: CNN kernel size (odd; same-padded).
        num_lstm_layers: Number of stacked LSTM layers.
        mode: ``"grid"`` (``[B,T,C,H,W] -> [B,P,H,W]``) or ``"point"``
            (``[B,T,C,H,W]`` patches ``-> [B,P]``).

    Returns:
        An initialised CNN-LSTM ``nn.Module`` honouring the selected ``mode``.

    Raises:
        ValueError: If ``mode`` is not ``"grid"`` or ``"point"``.
    """
    if mode not in ("grid", "point"):
        raise ValueError(f"mode must be 'grid' or 'point', got {mode!r}")
    torch, nn, F = _torch_modules()

    class CNNEncoder(nn.Module):
        """Stack of Conv-BN-ReLU blocks; spatially fully-convolutional (no pool)."""

        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            ch = in_channels
            pad = kernel_size // 2
            for out_ch in cnn_channels:
                layers += [
                    nn.Conv2d(ch, out_ch, kernel_size, padding=pad),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                ch = out_ch
            self.net = nn.Sequential(*layers)
            self.out_ch = ch

        def forward(self, x: "Tensor") -> "Tensor":
            return self.net(x)  # [B*T, ch, H, W]

    class CNNLSTMGrid(nn.Module):
        """Gridded CNN-LSTM: per-cell LSTM over CNN feature sequences.

        Forward: ``[B, T, C, H, W] -> [B, P, H, W]`` (final step).
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = CNNEncoder()
            self.lstm = nn.LSTM(
                input_size=self.encoder.out_ch,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            )
            self.head = nn.Conv2d(hidden_size, out_channels, 1)

        def forward(self, x: "Tensor") -> "Tensor":
            if x.dim() != 5:
                raise ValueError(f"expected 5-D [B,T,C,H,W], got {tuple(x.shape)}")
            b, t, c, h, w = x.shape
            # Encode every frame: fold T into batch for the CNN.
            feats = self.encoder(x.reshape(b * t, c, h, w))  # [B*T, ch, H, W]
            ch = feats.shape[1]
            feats = feats.reshape(b, t, ch, h, w)
            # Per-cell sequence -> LSTM. Move (H,W) into the batch dim.
            seq = feats.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, ch)
            out, _ = self.lstm(seq)  # [B*H*W, T, hidden]
            last = out[:, -1]  # [B*H*W, hidden]
            grid = last.reshape(b, h, w, hidden_size).permute(0, 3, 1, 2)
            return self.head(grid)  # [B, P, H, W]

    class CNNLSTMPoint(nn.Module):
        """Patch CNN-LSTM: pool each patch to a vector, LSTM over T, MLP head.

        Forward: ``[B, T, C, H, W]`` patches ``-> [B, P]``.
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = CNNEncoder()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.lstm = nn.LSTM(
                input_size=self.encoder.out_ch,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size // 2, out_channels),
            )

        def forward(self, x: "Tensor") -> "Tensor":
            if x.dim() != 5:
                raise ValueError(f"expected 5-D [B,T,C,H,W], got {tuple(x.shape)}")
            b, t, c, h, w = x.shape
            feats = self.encoder(x.reshape(b * t, c, h, w))
            vec = self.pool(feats).reshape(b, t, -1)  # [B, T, ch]
            out, _ = self.lstm(vec)  # [B, T, hidden]
            return self.head(out[:, -1])  # [B, P]

    model: Any = CNNLSTMGrid() if mode == "grid" else CNNLSTMPoint()
    _log.info(
        "Built CNN-LSTM (%s): in=%d out=%d cnn=%s hidden=%d.",
        mode, in_channels, out_channels, cnn_channels, hidden_size,
    )
    return model


__all__ = ["build_cnn_lstm"]
