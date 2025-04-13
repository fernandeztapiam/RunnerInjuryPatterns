# Proyecto: Descubrimiento de patrones en corredores principiantes (desde cero)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Paso 1: Simulación de datos realistas y variados (sin grupos explícitos)
np.random.seed(42)

def simulate_runners(n):
    stride = np.random.normal(loc=1.4, scale=0.15, size=n)         # longitud de zancada (1.1 a 1.7 m aprox)
    cadence = np.random.normal(loc=155, scale=12, size=n)          # pasos por minuto (130 a 180 aprox)
    forefoot = np.random.uniform(20, 40, size=n)                   # presión antepié
    midfoot = np.random.uniform(20, 35, size=n)                    # presión mediopié
    rearfoot = 100 - forefoot - midfoot                           # presión retropié, para que sume 100%

    return pd.DataFrame({
        "stride_length": stride,
        "cadence": cadence,
        "forefoot": forefoot,
        "midfoot": midfoot,
        "rearfoot": rearfoot
    })

# Generamos el dataset sin etiquetas
n_samples = 100
df = simulate_runners(n_samples)

# Exploración rápida de las primeras filas
print(df.head())
