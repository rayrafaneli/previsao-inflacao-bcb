"""Análise estatística da série: estacionariedade, decomposição, sazonalidade e autocorrelação."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def testes_estacionariedade(s: pd.Series) -> dict:
    """ADF (H0: raiz unitária) e KPSS (H0: estacionária). Concordância dá mais segurança na conclusão."""
    adf_stat, adf_p, *_ = adfuller(s.dropna(), autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s.dropna(), regression="c", nlags="auto")
    return {
        "adf_estatistica": round(float(adf_stat), 3), "adf_p": round(float(adf_p), 4),
        "kpss_estatistica": round(float(kpss_stat), 3), "kpss_p": round(float(kpss_p), 4),
        "estacionaria": bool(adf_p < 0.05 and kpss_p > 0.05),
    }


def decomposicao(s: pd.Series) -> pd.DataFrame:
    r = STL(s, period=12, robust=True).fit()
    return pd.DataFrame({"observado": s, "tendencia": r.trend, "sazonal": r.seasonal, "residuo": r.resid})


def forca_sazonal(dec: pd.DataFrame) -> float:
    """Força da sazonalidade (Hyndman): 1 - Var(resíduo) / Var(sazonal + resíduo)."""
    return float(max(0.0, 1 - dec["residuo"].var() / (dec["sazonal"] + dec["residuo"]).var()))


def sazonalidade_por_mes(s: pd.Series) -> pd.DataFrame:
    """Média por mês do ano com intervalo de confiança de 95% (t de Student)."""
    g = s.groupby(s.index.month)
    out = g.agg(["mean", "std", "count"])
    t = stats.t.ppf(0.975, out["count"] - 1)
    out["ic"] = t * out["std"] / np.sqrt(out["count"])
    return out


def autocorrelacao(s: pd.Series, lags: int = 24) -> pd.DataFrame:
    s = s.dropna()
    return pd.DataFrame({"acf": acf(s, nlags=lags)[1:], "pacf": pacf(s, nlags=lags, method="ywm")[1:]},
                        index=range(1, lags + 1))
