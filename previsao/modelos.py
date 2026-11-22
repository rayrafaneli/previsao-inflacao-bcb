"""Modelos de previsão um passo à frente, todos com a mesma interface: prever(historico) -> float."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

LAGS = (1, 2, 3, 6, 12)


def features(base: pd.DataFrame) -> pd.DataFrame:
    """Defasagens do IPCA, da Selic e do câmbio + mês do ano (seno/cosseno). Só usa informação passada."""
    X = pd.DataFrame(index=base.index)
    for l in LAGS:
        X[f"ipca_l{l}"] = base["ipca"].shift(l)
    X["ipca_mm12_l1"] = base["ipca"].shift(1).rolling(12).mean()
    X["selic_l1"] = base["selic"].shift(1)
    X["dolar_var_l1"] = base["dolar_var"].shift(1)
    X["dolar_var_l2"] = base["dolar_var"].shift(2)
    X["mes_sin"] = np.sin(2 * np.pi * base.index.month / 12)
    X["mes_cos"] = np.cos(2 * np.pi * base.index.month / 12)
    return X


class NaiveSazonal:
    nome = "Naive sazonal"

    def prever(self, base, alvo):
        return float(base.loc[alvo - pd.DateOffset(months=12), "ipca"])


class MediaMovel:
    nome = "Média móvel 12m"

    def prever(self, base, alvo):
        return float(base.loc[:alvo - pd.DateOffset(months=1), "ipca"].tail(12).mean())


class Sarima:
    nome = "SARIMA"
    ordem, sazonal = (1, 0, 1), (1, 0, 1, 12)

    def ajustar(self, serie: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(serie, order=self.ordem, seasonal_order=self.sazonal, trend="c",
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    def prever(self, base, alvo):
        hist = base.loc[:alvo - pd.DateOffset(months=1), "ipca"]
        return float(self.ajustar(hist).forecast(1).iloc[0])


class _Regressao:
    def _treino(self, base, alvo):
        X = features(base)
        treino = X.loc[:alvo - pd.DateOffset(months=1)].dropna()
        return treino, base.loc[treino.index, "ipca"], X.loc[[alvo]]


class Ridge(_Regressao):
    nome = "Regressão Ridge"

    def prever(self, base, alvo):
        X, y, x = self._treino(base, alvo)
        return float(RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X, y).predict(x)[0])


class XGBoost(_Regressao):
    nome = "XGBoost"

    def prever(self, base, alvo):
        X, y, x = self._treino(base, alvo)
        m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, n_jobs=1)
        return float(m.fit(X, y).predict(x)[0])


TODOS = [NaiveSazonal(), MediaMovel(), Sarima(), Ridge(), XGBoost()]
