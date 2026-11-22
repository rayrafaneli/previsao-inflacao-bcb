"""Gráficos do relatório no tema escuro (cores do Power BI)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FUNDO, PAINEL, GRADE, TEXTO, SUAVE = "#161514", "#252423", "#3B3A39", "#F3F2F1", "#A19F9D"
AMARELO, AZUL, LARANJA, ROXO, ROSA = "#F2C811", "#118DFF", "#E66C37", "#744EC2", "#E044A7"
CORES = {"Naive sazonal": SUAVE, "Média móvel 12m": ROSA, "SARIMA": AZUL, "Regressão Ridge": LARANJA, "XGBoost": ROXO, "Ensemble": "#1AAB40"}
TAM = (13.6, 8.33)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "text.color": TEXTO,
    "axes.facecolor": PAINEL, "axes.edgecolor": PAINEL, "axes.labelcolor": SUAVE, "axes.titlecolor": TEXTO,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
    "grid.color": GRADE, "grid.linestyle": "--", "grid.alpha": .6, "xtick.color": SUAVE, "ytick.color": SUAVE,
    "legend.frameon": False, "figure.facecolor": FUNDO, "savefig.facecolor": FUNDO,
})


def _titulo(fig, titulo, sub):
    fig.text(.012, .965, titulo, fontsize=20, weight="bold", color=TEXTO, va="top")
    fig.text(.012, .915, sub, fontsize=12, color=SUAVE, va="top")


def _salvar(fig, destino: Path):
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    return destino


def serie(dec: pd.DataFrame, testes: dict, forca: float, destino: Path):
    fig = plt.figure(figsize=TAM)
    _titulo(fig, "IPCA mensal · série e decomposição STL",
            f"ADF p = {testes['adf_p']:.3f} · KPSS p = {testes['kpss_p']:.3f} · força sazonal = {forca:.2f}")
    ax1 = fig.add_axes([.05, .42, .93, .43])
    ax1.bar(dec.index, dec["observado"], width=25, color=AMARELO, alpha=.85, label="IPCA no mês (%)")
    ax1.plot(dec.index, dec["tendencia"], color=AZUL, lw=2.6, label="tendência (STL)")
    ax1.axhline(0, color=GRADE, lw=1)
    ax1.legend(loc="upper right", ncols=2)
    ax2 = fig.add_axes([.05, .07, .45, .27])
    ax2.plot(dec.index, dec["sazonal"], color=LARANJA, lw=1.6)
    ax2.set_title("componente sazonal", fontsize=12)
    ax3 = fig.add_axes([.53, .07, .45, .27])
    ax3.bar(dec.index, dec["residuo"], width=25, color=ROXO, alpha=.8)
    ax3.set_title("resíduo", fontsize=12)
    return _salvar(fig, destino)


def sazonal_acf(saz: pd.DataFrame, ac: pd.DataFrame, n: int, destino: Path):
    fig = plt.figure(figsize=TAM)
    _titulo(fig, "Sazonalidade e autocorrelação", "média por mês com IC de 95% · ACF e PACF com banda de ±1,96/√n")
    ax = fig.add_axes([.05, .08, .44, .76])
    meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    ax.bar(meses, saz["mean"], color=AMARELO, yerr=saz["ic"], ecolor=TEXTO, capsize=4)
    ax.set_title("IPCA médio por mês do ano (%)")
    banda = 1.96 / np.sqrt(n)
    for i, (col, cor) in enumerate([("acf", AZUL), ("pacf", LARANJA)]):
        a = fig.add_axes([.55, .48 - i * .40, .43, .33])
        a.bar(ac.index, ac[col], color=cor, width=.6)
        a.axhspan(-banda, banda, color=SUAVE, alpha=.18)
        a.axhline(0, color=GRADE)
        a.set_title(col.upper(), fontsize=12)
        a.set_xticks([1, 6, 12, 18, 24])
    return _salvar(fig, destino)


def backtest(prev: pd.DataFrame, met: pd.DataFrame, desafiante: str, dm: tuple, destino: Path):
    fig = plt.figure(figsize=TAM)
    melhor = met.index[0]
    _titulo(fig, "Backtest walk-forward · 36 meses fora da amostra",
            f"menor RMSE: {melhor} · menor MAE: {met['MAE'].idxmin()} · "
            f"Diebold-Mariano {desafiante} × naive: DM = {dm[0]:.2f}, p = {dm[1]:.2f}")
    ax = fig.add_axes([.05, .08, .56, .76])
    ax.plot(prev.index, prev["real"], color=TEXTO, lw=2.8, label="IPCA real", marker="o", ms=3.5)
    for col in ["Ensemble", "Naive sazonal", "SARIMA"]:
        if col in prev:
            ax.plot(prev.index, prev[col], color=CORES[col], lw=2.2 if col == "Ensemble" else 1.4,
                    ls="-" if col == "Ensemble" else "--", label=col)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("previsto × real (%)")
    ax2 = fig.add_axes([.68, .08, .30, .76])
    ordem = met.iloc[::-1]
    barras = ax2.barh(ordem.index, ordem["RMSE"], color=[AMARELO if i == melhor else CORES[i] for i in ordem.index])
    ax2.bar_label(barras, labels=[f"{v:.3f}" for v in ordem["RMSE"]], padding=4, color=TEXTO)
    ax2.set_title("RMSE (p.p.) · menor é melhor")
    ax2.set_xlim(0, ordem["RMSE"].max() * 1.3)
    ax2.grid(axis="y", visible=False)
    return _salvar(fig, destino)


def previsao(hist: pd.Series, fc: pd.DataFrame, acum: float, destino: Path):
    fig = plt.figure(figsize=TAM)
    _titulo(fig, "Previsão dos próximos 12 meses · SARIMA",
            f"intervalos de 80% e 95% · IPCA acumulado projetado em 12 meses: {acum:.2f}%".replace(".", ","))
    ax = fig.add_axes([.05, .08, .93, .76])
    h = hist.tail(48)
    ax.bar(h.index, h.values, width=25, color=AMARELO, alpha=.85, label="IPCA realizado")
    ax.fill_between(fc.index, fc["lo95"], fc["hi95"], color=AZUL, alpha=.18, label="IC 95%")
    ax.fill_between(fc.index, fc["lo80"], fc["hi80"], color=AZUL, alpha=.32, label="IC 80%")
    ax.plot(fc.index, fc["media"], color=AZUL, lw=2.6, marker="o", ms=4, label="previsão")
    ax.axvline(h.index[-1] + pd.Timedelta(days=15), color=SUAVE, ls=":", lw=1.5)
    ax.axhline(0, color=GRADE)
    ax.legend(loc="upper left", ncols=4)
    ax.set_title("IPCA mensal (%)")
    return _salvar(fig, destino)
