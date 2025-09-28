from pathlib import Path
import pandas as pd
from modelos import treinar_rf
from AvaliarModelo import avaliar_modelo
from PrevisaoFinal import gerar_previsao

# --- Criar pasta data se não existir ---
Path("data").mkdir(exist_ok=True)

# --- Carregar dados ---
df = pd.read_csv("data/ipca_continuo.csv", index_col=0, parse_dates=True)

# --- Treinar modelo até 2020 ---
rf_model, X_train, y_train = treinar_rf(df, ate_ano='2020-12-01')

# --- Avaliação histórica 2021-2024 ---
val = df['2021-01-01':'2024-12-01']
X_val = val[[f'lag_{lag}' for lag in [1,3,6,12]]]
y_val = val['valor']

y_pred = rf_model.predict(X_val)

avaliar_modelo(y_val, y_pred, nome_modelo="Random Forest (2021-2024)")

# --- Previsão +6 anos ---
# Treinar com dados completos até 2024
rf_model, X_full, y_full = treinar_rf(df, ate_ano='2024-12-01')
future_df = gerar_previsao(df, rf_model)
