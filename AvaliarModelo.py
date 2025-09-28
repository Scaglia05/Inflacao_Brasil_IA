import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def avaliar_modelo(y_real, y_pred, nome_modelo="Modelo", plot=True):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False)
    
    print(f"=== Avaliação: {nome_modelo} ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("===============================")
    
    df = pd.DataFrame({"Real": y_real, "Previsto": y_pred}, index=y_real.index)
    df.to_csv("data/comparacao_modelo.csv")
    
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(y_real.index, y_real, label='Real', color='blue')
        plt.plot(y_real.index, y_pred, label='Previsto', color='red', linestyle='--')
        plt.title(f"{nome_modelo}: Real vs Previsto")
        plt.xlabel("Data")
        plt.ylabel("Inflação (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
