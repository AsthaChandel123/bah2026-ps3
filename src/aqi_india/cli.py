"""Typer command-line interface for aqi_india.

The root app ``aqi`` composes one sub-Typer per pipeline stage (ingest, fuse,
features, train, maps, hotspots, transport, validate, serve, demo) plus a
top-level ``aqi info`` command. Every command is importable and runnable today;
stage commands whose heavy implementation lands in later phases print an
informative "wired in later" message via rich rather than failing, so
``aqi --help`` and smoke tests always work.

Heavy dependencies (torch, earthengine, ...) are never imported here.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aqi_india import __version__

console = Console()

app = typer.Typer(
    name="aqi",
    help="Surface AQI mapping and HCHO hotspot detection over India (BAH 2026 PS3).",
    no_args_is_help=True,
    add_completion=False,
)

# One sub-Typer per pipeline stage.
ingest_app = typer.Typer(help="Ingest satellite / reanalysis / ground sources.", no_args_is_help=True)
fuse_app = typer.Typer(help="Fuse and gap-fill multi-source fields.", no_args_is_help=True)
features_app = typer.Typer(help="Build H3-keyed physics / met / static features.", no_args_is_help=True)
train_app = typer.Typer(help="Train Objective-1 surface-concentration models.", no_args_is_help=True)
maps_app = typer.Typer(help="Render daily surface AQI maps.", no_args_is_help=True)
hotspots_app = typer.Typer(help="Detect HCHO hotspots (Objective-2).", no_args_is_help=True)
transport_app = typer.Typer(help="Fire-HCHO correlation and transport analysis.", no_args_is_help=True)
validate_app = typer.Typer(help="Validate predictions against CPCB (CV ladder).", no_args_is_help=True)
serve_app = typer.Typer(help="Serve interactive / static map products.", no_args_is_help=True)
demo_app = typer.Typer(help="Run the light synthetic end-to-end demo.", no_args_is_help=True)

app.add_typer(ingest_app, name="ingest")
app.add_typer(fuse_app, name="fuse")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(maps_app, name="maps")
app.add_typer(hotspots_app, name="hotspots")
app.add_typer(transport_app, name="transport")
app.add_typer(validate_app, name="validate")
app.add_typer(serve_app, name="serve")
app.add_typer(demo_app, name="demo")


def _wired_later(stage: str, detail: str) -> None:
    """Print a uniform 'implemented in a later phase' notice."""
    console.print(
        Panel(
            f"[yellow]{detail}[/yellow]\n\n"
            f"[dim]This [b]{stage}[/b] command is part of the project spine and will be "
            f"wired to its full implementation by the module agents. The core engine "
            f"(NAQI, H3 fusion key, metrics) is already live and unit-tested.[/dim]",
            title=f"aqi {stage}",
            border_style="cyan",
        )
    )


# --------------------------------------------------------------------------- #
# Top-level info                                                              #
# --------------------------------------------------------------------------- #
@app.command()
def info() -> None:
    """Print project name, version and the two PS3 objectives."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[b]Project[/b]", "aqi-india — BAH 2026 (ISRO) Problem Statement 3")
    table.add_row("[b]Version[/b]", __version__)
    table.add_row(
        "[b]Objective 1[/b]",
        "Daily surface AQI maps over India from satellite columns + INSAT-3D AOD\n"
        "+ reanalysis meteorology, learned against CPCB (CNN/LSTM/CNN-LSTM),\n"
        "scored by RMSE / R / MAE.",
    )
    table.add_row(
        "[b]Objective 2[/b]",
        "High-resolution HCHO hotspot detection during biomass-burning seasons,\n"
        "fire-HCHO correlation, and transport attribution (IGP + forest belts).",
    )
    console.print(Panel(table, title="aqi info", border_style="green"))


# --------------------------------------------------------------------------- #
# ingest                                                                       #
# --------------------------------------------------------------------------- #
@ingest_app.command("s5p")
def ingest_s5p(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., help="End date YYYY-MM-DD."),
) -> None:
    """Ingest Sentinel-5P TROPOMI L3 columns (NO2/SO2/CO/O3/HCHO/AER_AI)."""
    _wired_later("ingest s5p", f"Would ingest S5P L3 columns for {start}..{end} via GEE.")


@ingest_app.command("cpcb")
def ingest_cpcb(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., help="End date YYYY-MM-DD."),
) -> None:
    """Ingest CPCB CAAQMS ground-truth pollutant labels."""
    _wired_later("ingest cpcb", f"Would pull CPCB station data for {start}..{end}.")


# --------------------------------------------------------------------------- #
# fuse                                                                          #
# --------------------------------------------------------------------------- #
@fuse_app.command("gapfill")
def fuse_gapfill() -> None:
    """Build the seamless daily cube (DINEOF -> U-Net inpaint -> kriging)."""
    _wired_later("fuse gapfill", "Would run the gap-fill cascade and bias-correction.")


# --------------------------------------------------------------------------- #
# features                                                                     #
# --------------------------------------------------------------------------- #
@features_app.command("build")
def features_build() -> None:
    """Assemble the H3-keyed physics / met / static feature matrix."""
    _wired_later("features build", "Would build the [B,T,C,H,W] tensors + GeoParquet matrix.")


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #
@train_app.command("saconvlstm")
def train_saconvlstm() -> None:
    """Train the physics-guided SA-ConvLSTM surface-concentration model."""
    _wired_later("train saconvlstm", "Would train SA-ConvLSTM (requires the 'deep' extra).")


@train_app.command("lightgbm")
def train_lightgbm() -> None:
    """Train the LightGBM tabular baseline / stacking member."""
    _wired_later("train lightgbm", "Would train the LightGBM column->surface model.")


# --------------------------------------------------------------------------- #
# maps                                                                         #
# --------------------------------------------------------------------------- #
@maps_app.command("daily")
def maps_daily() -> None:
    """Render the daily surface AQI map (concentration grid -> NAQI -> COG)."""
    _wired_later("maps daily", "Would render daily India AQI COGs + responsible-pollutant.")


# --------------------------------------------------------------------------- #
# hotspots                                                                     #
# --------------------------------------------------------------------------- #
@hotspots_app.command("detect")
def hotspots_detect() -> None:
    """Detect HCHO hotspots (Gi* + LISA + percentile consensus)."""
    _wired_later("hotspots detect", "Would run the >=2-of-3 hotspot consensus + EHSA.")


# --------------------------------------------------------------------------- #
# transport                                                                    #
# --------------------------------------------------------------------------- #
@transport_app.command("hysplit")
def transport_hysplit() -> None:
    """Run HYSPLIT back/forward trajectories and trajectory clustering."""
    _wired_later("transport hysplit", "Would run ERA5-driven HYSPLIT + CWT/PSCF.")


# --------------------------------------------------------------------------- #
# validate                                                                     #
# --------------------------------------------------------------------------- #
@validate_app.command("run")
def validate_run() -> None:
    """Run the CV ladder and emit the validation report."""
    _wired_later("validate run", "Would run LOSO + spatiotemporal-blocked CV + confusion.")


# --------------------------------------------------------------------------- #
# serve                                                                        #
# --------------------------------------------------------------------------- #
@serve_app.command("dashboard")
def serve_dashboard() -> None:
    """Launch the interactive Streamlit + TiTiler dashboard."""
    _wired_later("serve dashboard", "Would launch Streamlit (requires the 'serve' extra).")


# --------------------------------------------------------------------------- #
# demo                                                                         #
# --------------------------------------------------------------------------- #
@demo_app.command("run")
def demo_run() -> None:
    """Run the light synthetic end-to-end demo (no credentialed deps)."""
    _wired_later(
        "demo run",
        "Would synthesize a tiny India grid + CPCB stations + fires and run the "
        "full pipeline (fusion -> features -> model -> NAQI maps -> hotspots) on "
        "the light dependency set.",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
