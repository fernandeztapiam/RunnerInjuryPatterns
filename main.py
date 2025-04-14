import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Generación de datos simulados
np.random.seed(42)
n_samples = 150

stride_length = np.random.normal(loc=1.2, scale=0.15, size=n_samples)         # más variabilidad
cadence = np.random.normal(loc=155, scale=10, size=n_samples)
plantar_pressure = np.random.normal(loc=350, scale=40, size=n_samples)

data = pd.DataFrame({
    'Stride Length (m)': stride_length,
    'Cadence (spm)': cadence,
    'Plantar Pressure (kPa)': plantar_pressure
})

# 2. Análisis exploratorio - Histogramas
sns.set(style="whitegrid")
data.hist(bins=20, figsize=(10, 5), color='skyblue', edgecolor='black')
plt.suptitle("Distribución de Variables Simuladas", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. Boxplot tradicional (sin normalizar)
plt.figure(figsize=(10, 4))
sns.boxplot(data=data, palette="pastel")
plt.title("Boxplots de Variables Simuladas")
plt.show()

# 4. Pairplot (relaciones entre variables)
sns.pairplot(data)
plt.suptitle("Relaciones entre Variables Simuladas", y=1.02)
plt.show()

# 5. Visualización con datos normalizados (solo para comparar escalas)
data_norm = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)

plt.figure(figsize=(10, 4))
sns.boxplot(data=data_norm, palette="pastel")
plt.title("Boxplots Normalizados de Variables Simuladas")
plt.show()

# 6. Matriz de correlación de Pearson
correlation_matrix = data.corr()
print("\n🔹 Matriz de correlación de Pearson:")
print(correlation_matrix.round(2))

# Visualización de la matriz de correlación
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Variables")
plt.tight_layout()
plt.show()

import scipy.stats as stats

# 7. Regresión lineal: Cadence ~ Stride Length
x = data['Stride Length (m)']
y = data['Cadence (spm)']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\n🔹 Regresión lineal: Cadence ~ Stride Length")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R²: {r_value**2:.2f}  (correlación = {r_value:.2f})")
print(f"p-value: {p_value:.4f}")

# Gráfico con línea de regresión
plt.figure(figsize=(6, 4))
sns.scatterplot(x=x, y=y)
plt.plot(x, intercept + slope * x, color='red', label=f'Regresión lineal\n$R^2$ = {r_value**2:.2f}')
plt.xlabel("Stride Length (m)")
plt.ylabel("Cadence (spm)")
plt.title("Relación entre Zancada y Cadencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
