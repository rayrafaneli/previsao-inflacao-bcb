"""Coleta das séries do SGS/BCB e montagem da base mensal usada na modelagem."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
CACHE = Path(__file__).resolve().parent.parent / "data"
INICIO = date(2004, 1, 1)

# código SGS -> (coluna, como agregar no mês)
SERIES = {
    433: ("ipca", "ultimo"),       # IPCA, variação % no mês (alvo)
    432: ("selic", "ultimo"),      # meta Selic, % a.a. (valor de fim de mês)
    1: ("dolar", "media"),         # dólar comercial venda, R$ (média do mês)
}

log = logging.getLogger(__name__)


def _janelas(inicio: date, fim: date, anos: int = 5):
    atual = inicio
    while atual <= fim:
        prox = min(date(atual.year + anos, atual.month, atual.day) - timedelta(days=1), fim)
        yield atual, prox
        atual = prox + timedelta(days=1)


def baixar(codigo: int, fim: date | None = None) -> pd.DataFrame:
    registros = []
    for a, b in _janelas(INICIO, fim or date.today()):
        for tentativa in range(4):
            try:
                r = requests.get(API.format(codigo=codigo), timeout=60,
                                 params={"formato": "json", "dataInicial": f"{a:%d/%m/%Y}", "dataFinal": f"{b:%d/%m/%Y}"})
                r.raise_for_status()
                registros += r.json()
                break
            except (requests.RequestException, ValueError):
                if tentativa == 3:
                    raise
                time.sleep(2 ** (tentativa + 1))
    df = pd.DataFrame(registros)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna()


def mensal(df: pd.DataFrame, agregacao: str) -> pd.Series:
    s = df.drop_duplicates("data").set_index("data")["valor"].sort_index().resample("MS")
    return (s.last() if agregacao == "ultimo" else s.mean()).dropna()


def base_mensal(usar_cache: bool = True) -> pd.DataFrame:
    """DataFrame mensal com ipca, selic, dolar e a variação % mensal do dólar."""
    CACHE.mkdir(exist_ok=True)
    arquivo = CACHE / "base_mensal.csv"
    if usar_cache and arquivo.exists():
        return pd.read_csv(arquivo, parse_dates=["data"], index_col="data").asfreq("MS")
    colunas = {}
    for codigo, (nome, agg) in SERIES.items():
        colunas[nome] = mensal(baixar(codigo), agg)
        log.info("%-6s %d meses", nome, len(colunas[nome]))
    base = pd.DataFrame(colunas).asfreq("MS")
    base["dolar_var"] = base["dolar"].pct_change() * 100
    base = base.dropna(subset=["ipca"])
    base.index.name = "data"
    base.to_csv(arquivo)
    return base
