"""Roda a análise completa: dados -> estatística -> backtest de 5 modelos -> previsão -> relatório em docs/.

Uso: python run.py [--atualizar]   (--atualizar baixa as séries de novo em vez de usar o cache)
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from previsao import estatistica as est
from previsao import graficos
from previsao.backtest import dm_contra, metricas, walk_forward
from previsao.dados import base_mensal
from previsao.modelos import TODOS, Sarima

DOCS = Path(__file__).resolve().parent / "docs"
log = logging.getLogger("previsao")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atualizar", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    t0 = time.perf_counter()
    DOCS.mkdir(exist_ok=True)

    base = base_mensal(usar_cache=not args.atualizar)
    ipca = base["ipca"]
    log.info("base: %d meses (%s a %s)", len(base), f"{base.index[0]:%m/%Y}", f"{base.index[-1]:%m/%Y}")

    testes = est.testes_estacionariedade(ipca)
    dec = est.decomposicao(ipca)
    forca = est.forca_sazonal(dec)
    log.info("ADF p=%.4f · KPSS p=%.4f · força sazonal=%.2f", testes["adf_p"], testes["kpss_p"], forca)

    prev = walk_forward(base, TODOS, meses_teste=36)
    met = metricas(prev)
    melhor = met.index[0]
    dms = dm_contra(prev)
    desafiante = next(m for m in met.index if m != "Naive sazonal")
    dm = tuple(dms.loc[desafiante, ["DM", "p"]])
    log.info("RMSE: %s", " · ".join(f"{m} {v:.3f}" for m, v in met["RMSE"].items()))
    log.info("menor RMSE: %s · menor MAE: %s · DM %s × naive = %.2f (p=%.3f)",
             melhor, met["MAE"].idxmin(), desafiante, *dm)

    ajuste = Sarima().ajustar(ipca)
    f95 = ajuste.get_forecast(12).summary_frame(alpha=.05)
    f80 = ajuste.get_forecast(12).summary_frame(alpha=.20)
    fc = pd.DataFrame({"media": f95["mean"], "lo95": f95["mean_ci_lower"], "hi95": f95["mean_ci_upper"],
                       "lo80": f80["mean_ci_lower"], "hi80": f80["mean_ci_upper"]})
    acum = (np.prod(1 + fc["media"] / 100) - 1) * 100
    log.info("IPCA projetado para os próximos 12 meses: %.2f%%", acum)

    graficos.serie(dec, testes, forca, DOCS / "1-serie.png")
    graficos.sazonal_acf(est.sazonalidade_por_mes(ipca), est.autocorrelacao(ipca), len(ipca), DOCS / "2-sazonalidade.png")
    graficos.backtest(prev, met, desafiante, dm, DOCS / "3-backtest.png")
    graficos.previsao(ipca, fc, acum, DOCS / "4-previsao.png")

    resultados = {
        "periodo": [f"{base.index[0]:%Y-%m}", f"{base.index[-1]:%Y-%m}"], "testes": testes,
        "forca_sazonal": round(forca, 3), "melhor_modelo": melhor,
        "metricas": met.round(4).reset_index().to_dict(orient="records"),
        "diebold_mariano_vs_naive": dms.round(4).reset_index().to_dict(orient="records"),
        "previsao_12m": {f"{d:%Y-%m}": round(v, 3) for d, v in fc["media"].items()},
        "ipca_acumulado_12m_projetado": round(acum, 2),
    }
    (DOCS / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    tabela = met.round(3).to_markdown()
    (DOCS / "resultados.md").write_text(
        f"# Resultados\n\nPeríodo: {resultados['periodo'][0]} a {resultados['periodo'][1]}\n\n"
        f"## Backtest (36 meses, um passo à frente)\n\n{tabela}\n\n"
        f"## Diebold-Mariano contra o naive sazonal\n\nDM < 0 = erra menos que o naive; p < 0,05 = diferença significativa.\n\n"
        f"{dms.round(3).to_markdown()}\n\n"
        f"## Previsão SARIMA\n\nIPCA acumulado projetado para os próximos 12 meses: **{acum:.2f}%**\n",
        encoding="utf-8")
    log.info("relatório em docs/ · %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
