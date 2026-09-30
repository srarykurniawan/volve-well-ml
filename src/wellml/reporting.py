"""Figur laporan statis (PNG) — palet tervalidasi CVD, satu sumbu, mark tipis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BAND = "#b7d3f6"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False,
})


def production_forecast(te, out: Path):
    wells = sorted(te["well"].unique())
    fig, axes = plt.subplots(len(wells), 1, figsize=(9, 2.1 * len(wells)), sharex=True)
    for ax, w in zip(np.atleast_1d(axes), wells):
        g = te[te["well"] == w].sort_values("date")
        ax.fill_between(g["date"], g["p10"], g["p90"], color=BAND, lw=0, label="P10–P90")
        ax.plot(g["date"], g["target"], color=INK, lw=1.4, label="Aktual (rata-rata 30 hari ke depan)")
        ax.plot(g["date"], g["pred_lgbm"], color=SERIES[0], lw=2, label="LightGBM P50")
        ax.plot(g["date"], g["pred_persistence"], color=SERIES[1], lw=1.2, ls="--", label="Persistence")
        ax.set_title(w, fontsize=9.5)
        ax.set_ylabel("Sm³/hari")
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Forecast produksi minyak 30 hari — holdout out-of-time Volve", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def calibration(cal_rows, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="Kalibrasi sempurna")
    ax.plot([r["pred"] for r in cal_rows], [r["obs"] for r in cal_rows], color=SERIES[0], lw=2, marker="o", ms=5,
            label="Model")
    ax.set_xlabel("Probabilitas prediksi")
    ax.set_ylabel("Frekuensi aktual")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def pred_vs_actual(y, model, afe, out: Path, title: str, unit: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    lim = [0, max(np.max(y), np.max(model), np.max(afe)) * 1.05]
    ax.plot(lim, lim, color=AXIS, lw=1, ls="--")
    ax.scatter(y, afe, s=10, color=SERIES[1], alpha=0.55, label="Estimasi AFE", lw=0)
    ax.scatter(y, model, s=10, color=SERIES[0], alpha=0.7, label="Model ML", lw=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Aktual ({unit})")
    ax.set_ylabel(f"Prediksi ({unit})")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def shap_bar(shap_rows, out: Path, title: str):
    rows = shap_rows[:12][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.barh([r["feature"] for r in rows], [r["mean_abs_shap"] for r in rows], color=SERIES[0], height=0.6)
    ax.set_xlabel("Rata-rata |SHAP|")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
