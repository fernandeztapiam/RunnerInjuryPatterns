import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Paso 1: Simulación de datos
np.random.seed(42)

def simulate_group(n, stride_mean, stride_sd, cadence_mean, cadence_sd, forefoot_range, midfoot_range):
    stride = np.random.normal(loc=stride_mean, scale=stride_sd, size=n)
    cadence = np.random.normal(loc=cadence_mean, scale=cadence_sd, size=n)
    forefoot = np.random.uniform(*forefoot_range, size=n)
    midfoot = np.random.uniform(*midfoot_range, size=n)
    rearfoot = 100 - forefoot - midfoot
    return pd.DataFrame({
        "stride_length": stride,
        "cadence": cadence,
        "forefoot": forefoot,
        "midfoot": midfoot,
        "rearfoot": rearfoot
    })

group_a = simulate_group(30, 1.2, 0.1, 150, 5, (20, 30), (20, 30))  # Beginner - weak technique
group_b = simulate_group(30, 1.6, 0.1, 145, 5, (25, 35), (20, 30))  # Beginner - overstriding
group_c = simulate_group(40, 1.4, 0.05, 165, 5, (30, 36), (30, 36)) # Efficient

df = pd.concat([group_a, group_b, group_c], ignore_index=True)

# Paso 2: Verificación de variabilidad
plt.figure(figsize=(12, 6))
for i, col in enumerate(df.columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} (std = {df[col].std():.2f})")
plt.tight_layout()
plt.show()

# Paso 3: Correlación entre variables
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Paso 4: Escalado
features = df.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Paso 5: Clustering (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Paso 6: PCA + visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="Set1")
plt.title("PCA of Runner Profiles with K-Means Clusters")
plt.grid(True)
plt.show()

# Paso 7: Análisis de clusters
cluster_means = df.groupby("cluster")[["stride_length", "cadence", "forefoot", "midfoot", "rearfoot"]].mean()
print("Cluster Means:\n", cluster_means)

# Paso 8: Regresión para explorar relaciones
X_reg = df[["stride_length"]]
y_reg = df["cadence"]
model = LinearRegression().fit(X_reg, y_reg)
r2 = model.score(X_reg, y_reg)

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["stride_length"], y=df["cadence"])
plt.plot(df["stride_length"], model.predict(X_reg), color='red')
plt.title(f"Stride Length vs Cadence (R² = {r2:.2f})")
plt.xlabel("Stride Length (m)")
plt.ylabel("Cadence (spm)")
plt.grid(True)
plt.show()

# Paso 9: Conclusión
print(f"Linear regression R² between stride length and cadence: {r2:.2f}")
