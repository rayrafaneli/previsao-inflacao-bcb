import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from previsao import estatistica as est  # noqa: E402
from previsao.backtest import diebold_mariano, metricas, walk_forward  # noqa: E402
from previsao.modelos import MediaMovel, NaiveSazonal, Ridge, features  # noqa: E402


@pytest.fixture
def base():
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(0)
    sazonal = 0.3 * np.sin(2 * np.pi * idx.month / 12)
    ipca = 0.4 + sazonal + rng.normal(0, 0.05, len(idx))
    return pd.DataFrame({"ipca": ipca, "selic": 10.0, "dolar": 5.0, "dolar_var": rng.normal(0, 1, len(idx))}, index=idx)


def test_features_usam_so_o_passado(base):
    X = features(base)
    alvo = base.index[50]
    assert X.loc[alvo, "ipca_l1"] == base["ipca"].iloc[49]
    assert X.loc[alvo, "ipca_l12"] == base["ipca"].iloc[38]
    base2 = base.copy()
    base2.loc[alvo, "ipca"] = 999  # mudar o mês-alvo não pode mudar as features dele
    assert features(base2).loc[alvo].equals(X.loc[alvo])


def test_naive_sazonal_e_media_movel(base):
    alvo = base.index[-1]
    assert NaiveSazonal().prever(base, alvo) == base["ipca"].iloc[-13]
    assert MediaMovel().prever(base, alvo) == pytest.approx(base["ipca"].iloc[-13:-1].mean())


def test_walk_forward_sem_vazamento(base):
    class Espiao:
        nome = "espião"
        vistos = []

        def prever(self, b, alvo):
            Espiao.vistos.append((alvo, b.loc[:alvo - pd.DateOffset(months=1)].index.max()))
            return 0.0

    prev = walk_forward(base, [Espiao()], meses_teste=6)
    assert len(prev) == 6
    assert all(ultimo < alvo for alvo, ultimo in Espiao.vistos)


def test_ridge_aprende_sazonalidade(base):
    prev = walk_forward(base, [NaiveSazonal(), Ridge()], meses_teste=12)
    met = metricas(prev)
    assert met.loc["Regressão Ridge", "RMSE"] < 0.15


def test_diebold_mariano():
    rng = np.random.default_rng(1)
    bom, ruim = pd.Series(rng.normal(0, .1, 60)), pd.Series(rng.normal(0, .5, 60))
    dm, p = diebold_mariano(bom, ruim)
    assert dm < 0 and p < 0.01
    a, b = pd.Series(rng.normal(0, .3, 60)), pd.Series(rng.normal(0, .3, 60))
    dm, p = diebold_mariano(a, b)  # mesma distribuição de erro: não deve rejeitar H0
    assert p > 0.05


def test_estatistica(base):
    t = est.testes_estacionariedade(base["ipca"])
    assert t["estacionaria"]
    dec = est.decomposicao(base["ipca"])
    assert est.forca_sazonal(dec) > 0.8
    saz = est.sazonalidade_por_mes(base["ipca"])
    assert list(saz.index) == list(range(1, 13)) and (saz["ic"] > 0).all()
