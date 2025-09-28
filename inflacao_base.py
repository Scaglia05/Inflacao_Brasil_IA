import requests
import pandas as pd
from pathlib import Path

# Série IPCA mensal (%), código 433 no SGS
series_id = 433
start = "01/01/2000"
end   = "31/12/2025"

url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start}&dataFinal={end}"

# Request
r = requests.get(url, timeout=30)
r.raise_for_status()
data = pd.DataFrame(r.json())

# Ajustes
data['data'] = pd.to_datetime(data['data'], dayfirst=True)
data['valor'] = pd.to_numeric(data['valor'].str.replace(',', '.'))
data = data.sort_values('data').set_index('data')

# Criar pasta para salvar
out = Path("data")
out.mkdir(exist_ok=True)

# Criar índice mensal contínuo
full_index = pd.date_range(start='2000-01-01', end=data.index.max(), freq='MS')
data = data.reindex(full_index)

# Preencher valores faltantes
data['valor'] = data['valor'].fillna(method='ffill')

# Salvar CSV atualizado
data.to_csv(out/"ipca_continuo.csv")

# Checar primeiros e últimos valores
print(data.head(12))
print(data.tail(12))

# Mostrar meses originalmente sem dados
missing = data[data['valor'].isna()]
print(f"Meses sem dados originais: {len(missing)}")
print(missing)
