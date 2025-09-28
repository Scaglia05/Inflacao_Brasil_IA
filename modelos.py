import pandas as pd
from sklearn.ensemble import RandomForestRegressor

LAGS = [1, 3, 6, 12]

def treinar_rf(df, ate_ano='2020-12-01'):
    """
    Treina Random Forest com dados até 'ate_ano'.
    Retorna: modelo treinado, X_train, y_train
    """
    # Criar lags
    for lag in LAGS:
        df[f'lag_{lag}'] = df['valor'].shift(lag)
    
    df = df.dropna()
    
    train = df[:ate_ano]
    X_train = train[[f'lag_{lag}' for lag in LAGS]]
    y_train = train['valor']
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf, X_train, y_train
