"""Validação walk-forward (janela expansiva, um passo à frente), métricas e teste de Diebold-Mariano."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


ENSEMBLE = ["Naive sazonal", "SARIMA", "Regressão Ridge", "XGBoost"]  # definido antes de olhar o teste


def walk_forward(base: pd.DataFrame, modelos, meses_teste: int = 36) -> pd.DataFrame:
    """Para cada mês de teste, cada modelo só enxerga o histórico até o mês anterior.
    Também calcula a combinação simples (média) dos modelos de ENSEMBLE."""
    alvos = base.index[-meses_teste:]
    linhas = []
    for alvo in alvos:
        linha = {"data": alvo, "real": float(base.loc[alvo, "ipca"])}
        for m in modelos:
            linha[m.nome] = m.prever(base, alvo)
        linhas.append(linha)
    prev = pd.DataFrame(linhas).set_index("data")
    if set(ENSEMBLE) <= set(prev.columns):
        prev["Ensemble"] = prev[ENSEMBLE].mean(axis=1)
    return prev


def metricas(prev: pd.DataFrame) -> pd.DataFrame:
    real = prev["real"]
    out = []
    for col in prev.columns.drop("real"):
        erro = prev[col] - real
        out.append({"modelo": col, "MAE": np.abs(erro).mean(), "RMSE": np.sqrt((erro ** 2).mean()),
                    "viés": erro.mean(), "acerto_direcao": (np.sign(prev[col].diff()) == np.sign(real.diff())).iloc[1:].mean()})
    return pd.DataFrame(out).set_index("modelo").sort_values("RMSE")


def diebold_mariano(e1: pd.Series, e2: pd.Series, h: int = 1) -> tuple[float, float]:
    """H0: mesma acurácia (perda quadrática). DM < 0 indica que o modelo 1 erra menos. Correção de Harvey et al."""
    d = (e1 ** 2 - e2 ** 2).dropna()
    n = len(d)
    dm = d.mean() / np.sqrt(d.var(ddof=1) / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(dm), df=n - 1)
    return float(dm), float(p)


def dm_contra(prev: pd.DataFrame, referencia: str = "Naive sazonal") -> pd.DataFrame:
    """Diebold-Mariano de cada modelo contra a referência."""
    e_ref = prev[referencia] - prev["real"]
    linhas = []
    for col in prev.columns.drop(["real", referencia]):
        dm, p = diebold_mariano(prev[col] - prev["real"], e_ref)
        linhas.append({"modelo": col, "DM": dm, "p": p})
    return pd.DataFrame(linhas).set_index("modelo")
