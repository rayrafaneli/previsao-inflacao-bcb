# Resultados

Período: 2004-01 a 2026-10

## Backtest (36 meses, um passo à frente)

| modelo          |   MAE |   RMSE |   viés |   acerto_direcao |
|:----------------|------:|-------:|-------:|-----------------:|
| Naive sazonal   | 0.218 |  0.263 |  0.005 |            0.657 |
| Ensemble        | 0.201 |  0.271 |  0.005 |            0.543 |
| XGBoost         | 0.237 |  0.304 | -0.013 |            0.343 |
| Regressão Ridge | 0.24  |  0.308 |  0.001 |            0.371 |
| Média móvel 12m | 0.24  |  0.314 | -0.006 |            0.343 |
| SARIMA          | 0.237 |  0.316 |  0.026 |            0.4   |

## Diebold-Mariano contra o naive sazonal

DM < 0 = erra menos que o naive; p < 0,05 = diferença significativa.

| modelo          |    DM |     p |
|:----------------|------:|------:|
| Média móvel 12m | 1.04  | 0.305 |
| SARIMA          | 1.082 | 0.287 |
| Regressão Ridge | 0.949 | 0.349 |
| XGBoost         | 0.959 | 0.344 |
| Ensemble        | 0.2   | 0.843 |

## Previsão SARIMA

IPCA acumulado projetado para os próximos 12 meses: **5.07%**
