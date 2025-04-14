import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 1. Generación de datos simulados
np.random.seed(42)
n_samples = 150

stride_length = np.random.normal(loc=1.2, scale=0.15, size=n_samples)
cadence = np.random.normal(loc=155, scale=10, size=n_samples)
plantar_pressure = np.random.normal(loc=350, scale=40, size=n_samples)

# Crear DataFrame
data = pd.DataFrame({
    'Stride Length (m)': stride_length,
    'Cadence (spm)': cadence,
    'Plantar Pressure (kPa)': plantar_pressure
})

# 2. Calcular velocidad (Stride Length × Cadence)
data['Speed (m/min)'] = data['Stride Length (m)'] * data['Cadence (spm)']

# 3. Histogramas de distribución
sns.set(style="whitegrid")
data.hist(bins=20, figsize=(12, 6), color='skyblue', edgecolor='black')
plt.suptitle("Distribución de Variables Simuladas", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 4. Boxplot sin normalizar
plt.figure(figsize=(10, 4))
sns.boxplot(data=data, palette="pastel")
plt.title("Boxplots de Variables Simuladas")
plt.show()

# 5. Pairplot (correlaciones visuales)
sns.pairplot(data)
plt.suptitle("Relaciones entre Variables Simuladas", y=1.02)
plt.show()

# 6. Boxplots normalizados (solo para visualización)
data_norm = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
plt.figure(figsize=(10, 4))
sns.boxplot(data=data_norm, palette="pastel")
plt.title("Boxplots Normalizados de Variables Simuladas")
plt.show()

# 7. Matriz de correlación
correlation_matrix = data.corr()
print("\n🔹 Matriz de correlación de Pearson:")
print(correlation_matrix.round(2))

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Variables")
plt.tight_layout()
plt.show()

# 8. Regresión lineal: Cadence ~ Stride Length
x = data['Stride Length (m)']
y = data['Cadence (spm)']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\n🔹 Regresión lineal: Cadence ~ Stride Length")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R²: {r_value**2:.2f}  (correlación = {r_value:.2f})")
print(f"p-value: {p_value:.4f}")

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

# 9. Correlación y regresión: Speed vs Plantar Pressure
cor = data[['Speed (m/min)', 'Plantar Pressure (kPa)']].corr().iloc[0, 1]
print(f"\n🔹 Correlación entre velocidad y presión plantar: r = {cor:.2f}")

plt.figure(figsize=(6, 4))
sns.regplot(x='Speed (m/min)', y='Plantar Pressure (kPa)', data=data,
            scatter_kws={"alpha":0.6}, line_kws={"color": "red"})
plt.title("Relación entre Velocidad y Presión Plantar")
plt.xlabel("Speed (m/min)")
plt.ylabel("Plantar Pressure (kPa)")
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 10. Estandarización de los datos
features = ['Stride Length (m)', 'Cadence (spm)', 'Plantar Pressure (kPa)', 'Speed (m/min)']
X = StandardScaler().fit_transform(data[features])

# 11. Método del codo para elegir k
inertias = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, inertias, marker='o')
plt.title("Método del Codo para determinar k")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Silhouette Score
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title("Silhouette Score para diferentes k")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. Aplicación de K-Means con el k elegido (por ejemplo, k=4)
k = 4  # puedes cambiarlo cuando veas los gráficos
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

data['Cluster'] = labels

# 14. Visualización con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df['Cluster'] = labels

plt.figure(figsize=(6, 5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60)
plt.title("Visualización de Clústeres (PCA 2D)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 15. Análisis por grupo
cluster_summary = data.groupby('Cluster')[features].mean().round(2)
print("\n🔹 Promedio de cada variable por clúster:")
print(cluster_summary)
