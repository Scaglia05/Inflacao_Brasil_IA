import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LAGS = [1,3,6,12]
HORIZONTE = 72  # 6 anos à frente

def gerar_previsao(df, modelo):
    # Criar lags se não existirem
    for lag in LAGS:
        if f'lag_{lag}' not in df.columns:
            df[f'lag_{lag}'] = df['valor'].shift(lag)
    
    df = df.dropna()
    
    # Últimos valores conhecidos para previsão iterativa
    last_known = df.iloc[-1][[f'lag_{lag}' for lag in LAGS]].values
    
    future_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=HORIZONTE, freq='MS')
    future_preds = []
    
    for _ in range(HORIZONTE):
        pred = modelo.predict(last_known.reshape(1,-1))[0]
        future_preds.append(pred)
        last_known = np.roll(last_known, -1)
        last_known[-1] = pred
    
    future_df = pd.DataFrame(future_preds, index=future_dates, columns=['valor'])
    
    # Plot histórico + previsão
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['valor'], label='Histórico', color='blue')
    plt.plot(future_df.index, future_df['valor'], label='Previsão +6 anos', color='red', linestyle='--')
    plt.title("IPCA Mensal: Histórico e Previsão +6 anos (Random Forest)")
    plt.xlabel("Ano")
    plt.ylabel("Inflação (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    out = Path("data")
    out.mkdir(exist_ok=True)
    future_df.to_csv(out/"ipca_previsao_ml.csv")
    
    print("Previsão +6 anos salva em data/ipca_previsao_ml.csv")
    
    return future_df
