"""SA-ConvLSTM — the production Objective-1 gridded model (PyTorch).

SA-ConvLSTM (Self-Attention ConvLSTM) is the **recommended core** for producing
spatially-coherent daily surface-AQI maps over India. It augments a ConvLSTM
(whose hidden states are 2-D feature maps, so the whole grid is predicted at once
and stays spatially coherent) with two attention mechanisms that fix the vanilla
ConvLSTM's weak long-range memory:

* **temporal self-attention** over the per-cell sequence of hidden states, so a
  cell can attend to the most relevant past days (e.g. a pre-monsoon dust day
  three days back) rather than only the immediately previous step, and
* **channel attention** (squeeze-and-excitation) that re-weights feature channels
  per step, emphasising the physics-guided channels (AOD/BLH, FNR, ...) that
  matter for the current regime.

The network maps an input sequence ``[B, T, C, H, W]`` (T daily frames of C
satellite/met/physics channels over an ``H x W`` grid) to a per-cell surface
field ``[B, P, H, W]`` (P pollutants) for the last frame, or to a full sequence
``[B, T, P, H, W]`` when ``return_sequence=True``.

**Importability:** ``torch`` is imported lazily inside the constructors/factory
so this module imports under the light dependency set. The demo uses LightGBM;
this model (and :mod:`aqi_india.models.cnn_lstm`) satisfies the PS3 CNN/LSTM
mandate and runs when the ``deep`` extra (torch) is installed.

Shape convention throughout: ``B`` batch, ``T`` time steps, ``C`` input channels,
``P`` output pollutants, ``H``/``W`` grid rows/cols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import Tensor, nn

_log = get_logger("models.saconvlstm")


def _torch_modules() -> tuple[Any, Any, Any]:
    """Lazily import and return ``(torch, nn, F)`` (torch.nn.functional).

    Raises:
        ImportError: With install guidance if torch is unavailable.
    """
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        raise ImportError(
            "PyTorch is required for SA-ConvLSTM. Install the deep extra: "
            "`pip install -e .[deep]`."
        ) from exc
    return torch, nn, F


def build_saconvlstm(
    in_channels: int,
    out_channels: int,
    *,
    hidden_channels: int = 64,
    kernel_size: int = 3,
    num_layers: int = 2,
    attn_dim: int = 32,
    return_sequence: bool = False,
) -> "nn.Module":
    """Factory constructing a ready-to-train :class:`SAConvLSTM` module.

    Defined as a factory (rather than a bare class at module scope) so that the
    ``nn.Module`` subclasses are only *defined* once torch is importable, keeping
    this module import-light. See :class:`SAConvLSTM` (built here) for the
    forward contract.

    Args:
        in_channels: Number of input channels ``C`` (satellite + met + physics).
        out_channels: Number of output pollutants ``P``.
        hidden_channels: ConvLSTM hidden-state channel count.
        kernel_size: Convolution kernel size (odd; same-padded).
        num_layers: Number of stacked ConvLSTM layers.
        attn_dim: Embedding dim of the temporal self-attention.
        return_sequence: If True, ``forward`` returns ``[B, T, P, H, W]``;
            otherwise ``[B, P, H, W]`` for the final step.

    Returns:
        An initialised :class:`SAConvLSTM` ``nn.Module``.
    """
    torch, nn, F = _torch_modules()

    class ConvLSTMCell(nn.Module):
        """A single ConvLSTM cell: gates are convolutions over ``[x, h]``.

        Hidden/cell states are 2-D maps ``[B, hidden, H, W]``, so spatial
        structure is preserved through time (the key property that yields
        spatially-coherent full-grid output).
        """

        def __init__(self, in_ch: int, hid_ch: int, k: int) -> None:
            super().__init__()
            self.hid_ch = hid_ch
            pad = k // 2
            # One conv produces all four gate stacks (i, f, o, g) at once.
            self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=pad)

        def forward(
            self, x: "Tensor", state: "tuple[Tensor, Tensor]"
        ) -> "tuple[Tensor, Tensor]":
            # x: [B, in_ch, H, W]; h, c: [B, hid_ch, H, W]
            h, c = state
            gates = self.conv(torch.cat([x, h], dim=1))
            i, f, o, g = torch.split(gates, self.hid_ch, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)
            return h_next, c_next

        def init_state(self, b: int, h: int, w: int, device: Any, dtype: Any):
            zeros = torch.zeros(b, self.hid_ch, h, w, device=device, dtype=dtype)
            return zeros, zeros.clone()

    class ChannelAttention(nn.Module):
        """Squeeze-and-excitation channel attention over a feature map.

        Global-average-pools each channel to a scalar, learns per-channel gates
        via a bottleneck MLP, and re-scales the channels — emphasising the
        physics-guided channels relevant to the current regime.
        """

        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            mid = max(channels // reduction, 4)
            self.fc1 = nn.Conv2d(channels, mid, 1)
            self.fc2 = nn.Conv2d(mid, channels, 1)

        def forward(self, x: "Tensor") -> "Tensor":
            # x: [B, C, H, W]
            s = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            s = torch.relu(self.fc1(s))
            s = torch.sigmoid(self.fc2(s))
            return x * s

    class TemporalSelfAttention(nn.Module):
        """Per-cell temporal self-attention over the hidden-state sequence.

        The sequence of ConvLSTM hidden maps ``[B, T, hid, H, W]`` is treated as a
        token sequence *per spatial cell*: queries/keys/values are 1x1-conv
        projections, attention is computed along ``T`` independently at each
        ``(H, W)``, letting a cell pull information from the most relevant past
        day rather than only the previous step.
        """

        def __init__(self, hid_ch: int, dim: int) -> None:
            super().__init__()
            self.q = nn.Conv2d(hid_ch, dim, 1)
            self.k = nn.Conv2d(hid_ch, dim, 1)
            self.v = nn.Conv2d(hid_ch, hid_ch, 1)
            self.scale = dim ** -0.5

        def forward(self, seq: "Tensor") -> "Tensor":
            # seq: [B, T, hid, H, W] -> attended [B, hid, H, W] (last-step query)
            b, t, ch, h, w = seq.shape
            flat = seq.reshape(b * t, ch, h, w)
            q = self.q(flat).reshape(b, t, -1, h, w)
            k = self.k(flat).reshape(b, t, -1, h, w)
            v = self.v(flat).reshape(b, t, ch, h, w)
            # Attention weights along time, per cell: [B, T(query)=1 last, T(key)]
            q_last = q[:, -1:]  # [B, 1, dim, H, W] query = final step
            # scores: einsum over the embedding dim d -> [B, 1, T, H, W]
            scores = (q_last.unsqueeze(2) * k.unsqueeze(1)).sum(dim=3) * self.scale
            attn = torch.softmax(scores, dim=2)  # over key time
            # weighted sum of values: [B, 1, hid, H, W] -> squeeze query dim
            out = (attn.unsqueeze(3) * v.unsqueeze(1)).sum(dim=2)
            return out[:, 0]

    class SAConvLSTM(nn.Module):
        """Self-Attention ConvLSTM for gridded surface-concentration sequences.

        Forward contract:
            input  ``x``: ``[B, T, C, H, W]`` (C = :paramref:`in_channels`),
            output     : ``[B, P, H, W]`` (final step) or ``[B, T, P, H, W]`` when
                         ``return_sequence`` is True (P = :paramref:`out_channels`).
        """

        def __init__(self) -> None:
            super().__init__()
            self.num_layers = num_layers
            self.return_sequence = return_sequence
            self.cells = nn.ModuleList()
            self.chan_attn = nn.ModuleList()
            ch_in = in_channels
            for _ in range(num_layers):
                self.cells.append(ConvLSTMCell(ch_in, hidden_channels, kernel_size))
                self.chan_attn.append(ChannelAttention(hidden_channels))
                ch_in = hidden_channels
            self.temporal_attn = TemporalSelfAttention(hidden_channels, attn_dim)
            # 1x1 head maps hidden channels -> pollutant outputs.
            self.head = nn.Conv2d(hidden_channels, out_channels, 1)

        def forward(self, x: "Tensor") -> "Tensor":
            # x: [B, T, C, H, W]
            if x.dim() != 5:
                raise ValueError(f"expected 5-D [B,T,C,H,W], got {tuple(x.shape)}")
            b, t, _c, h, w = x.shape
            device, dtype = x.device, x.dtype

            layer_input = x
            hidden_seq: "Tensor | None" = None
            for li in range(self.num_layers):
                cell = self.cells[li]
                attn = self.chan_attn[li]
                hstate, cstate = cell.init_state(b, h, w, device, dtype)
                steps = []
                for ti in range(t):
                    hstate, cstate = cell(layer_input[:, ti], (hstate, cstate))
                    steps.append(attn(hstate))  # channel-attended hidden
                hidden_seq = torch.stack(steps, dim=1)  # [B, T, hid, H, W]
                layer_input = hidden_seq

            assert hidden_seq is not None
            if self.return_sequence:
                outs = [self.head(hidden_seq[:, ti]) for ti in range(t)]
                return torch.stack(outs, dim=1)  # [B, T, P, H, W]
            # Temporal self-attention summarises the sequence for the final map.
            summary = self.temporal_attn(hidden_seq)  # [B, hid, H, W]
            return self.head(summary)  # [B, P, H, W]

    model = SAConvLSTM()
    _log.info(
        "Built SA-ConvLSTM: in=%d out=%d hidden=%d layers=%d.",
        in_channels, out_channels, hidden_channels, num_layers,
    )
    return model


__all__ = ["build_saconvlstm"]
