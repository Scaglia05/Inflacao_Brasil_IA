# Previsão de Inflação Brasileira com Machine Learning

Projeto em Python que realiza previsão da inflação mensal brasileira (IPCA) usando **Random Forest**, combinando técnicas de séries temporais e aprendizado de máquina. Fornece análise histórica, avaliação de desempenho e projeção futura de forma modular e replicável.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-orange)

## Funcionalidades

- **Coleta de dados**  
  - Importa séries históricas do IPCA do Banco Central ou CSV local.  
  - Período analisado: 2000 até o último mês disponível.

- **Pré-processamento**  
  - Criação de lags de 1, 3, 6 e 12 meses.  
  - Tratamento de valores ausentes e padronização da série.

- **Treinamento e avaliação histórica**  
  - Treina Random Forest com dados de 2000 a 2020.  
  - Avalia previsões para 2021–2024 usando métricas MAE e RMSE.  
  - Salva comparação real x previsto em CSV e gera gráficos.

- **Previsão futura**  
  - Re-treina o modelo com dados até 2024.  
  - Gera previsões iterativas para os próximos 6 anos (2025–2030).  
  - Plota histórico + projeção e salva resultados em CSV.

- **Visualizações**  
  - Séries históricas e previsões.  
  - Gráficos de autocorrelação e autocorrelação parcial.  
  - Comparação real vs previsto.

- **Modularidade**  
  - `modelos.py` → Treinamento Random Forest.  
  - `avaliacao.py` → Métricas, comparação e gráficos.  
  - `previsao_final.py` → Previsão iterativa +6 anos.  
  - `main.py` → Orquestra treino, avaliação e previsão.

## Tecnologias

- Python 3.x  
- Pandas, NumPy, Matplotlib  
- Scikit-learn (Random Forest)  
- Statsmodels (séries temporais)

## Requisitos
pip install pandas numpy matplotlib scikit-learn statsmodels


## Uso
- Coloque o CSV ipca_continuo.csv na pasta data/.
- Execute main.py para treinar, avaliar e gerar previsões futuras.
- Resultados salvos em data/:
- comparacao_modelo.csv → Avaliação histórica (2021–2024).
- ipca_previsao_ml.csv → Previsões para 2025–2030.

## Estrutura do Projeto
- `data/` → Contém CSVs de dados e previsões (`ipca_continuo.csv`, `ipca_previsao_ml.csv`, etc.)
- `modelos.py` → Treinamento do Random Forest
- `avaliacao.py` → Avaliação de desempenho e gráficos
- `previsao_final.py` → Geração de previsões iterativas
- `main.py` → Script principal que orquestra treino, avaliação e previsão                 # Script principal

