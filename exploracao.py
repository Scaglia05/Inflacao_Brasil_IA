import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Carregar série contínua
data = pd.read_csv("data/ipca_continuo.csv", index_col=0, parse_dates=True)

# Plot autocorrelação
plt.figure(figsize=(12,4))
plot_acf(data['valor'], lags=36)
plt.title("Autocorrelação do IPCA mensal (36 lags)")
plt.savefig("data/acf_ipca.png")
plt.show()
