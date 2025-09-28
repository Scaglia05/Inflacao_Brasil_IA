import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar série contínua
data = pd.read_csv("data/ipca_continuo.csv", index_col=0, parse_dates=True)

# Plotar série completa
plt.figure(figsize=(12,5))
plt.plot(data.index, data['valor'], label='IPCA Mensal')
plt.title("Série de IPCA Mensal (2000 até último disponível)")
plt.xlabel("Ano")
plt.ylabel("Inflação (%)")
plt.grid(True)
plt.legend()
plt.show()

# Decomposição aditiva (ou multiplicativa se variação muito grande)
decomp = seasonal_decompose(data['valor'], model='additive', period=12)  # period=12 para sazonalidade anual

# Plotar decomposição
decomp.plot()
plt.show()

# Estatísticas básicas
print("Média mensal:", round(data['valor'].mean(), 2))
print("Desvio padrão:", round(data['valor'].std(), 2))
print("Máximo:", data['valor'].max())
print("Mínimo:", data['valor'].min())
