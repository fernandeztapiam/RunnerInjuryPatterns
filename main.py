import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# 1. Data Generation
# This section generates simulated data for three variables related to human movement:
# 'Stride Length (m)', 'Cadence (spm)' (steps per minute), and 'Plantar Pressure (kPa)'.
# NumPy's random.normal function is used to create arrays of 150 data points for each variable,
# drawing from a normal distribution with specified means (loc) and standard deviations (scale).
np.random.seed(42)  # for reproducibility of the random data
n_samples = 150

stride_length = np.random.normal(loc=1.2, scale=0.15, size=n_samples)
cadence = np.random.normal(loc=155, scale=10, size=n_samples)
plantar_pressure = np.random.normal(loc=350, scale=40, size=n_samples)

# Create a Pandas DataFrame to store the generated data in a structured way.
data = pd.DataFrame({
    'Stride Length (m)': stride_length,
    'Cadence (spm)': cadence,
    'Plantar Pressure (kPa)': plantar_pressure
})

# 2. Speed Calculation
# A new column 'Speed (m/min)' is calculated by multiplying the 'Stride Length (m)' by the 'Cadence (spm)'.
# This assumes that speed is a direct product of stride length and the number of steps taken per minute.
data['Speed (m/min)'] = data['Stride Length (m)'] * data['Cadence (spm)']

# 3. Distribution Histograms
# This section uses Seaborn and Matplotlib to visualize the distribution of each of the simulated variables.
# Histograms are created with 20 bins to show the frequency of values within different ranges for each variable.
sns.set(style="whitegrid")  # Sets a clean style for the plots
data.hist(bins=20, figsize=(12, 6), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Simulated Variables", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
plt.show()

# 4. Unnormalized Boxplots
# Boxplots are generated to show the central tendency, spread, and potential outliers of the original, unscaled data for each variable.
plt.figure(figsize=(10, 4))
sns.boxplot(data=data, palette="pastel")  # 'pastel' provides a color palette for the boxes
plt.title("Boxplots of Simulated Variables")
plt.show()

# 5. Pairplot (Visual Correlations)
# A pairplot creates a matrix of scatter plots for all pairs of variables in the DataFrame,
# along with histograms on the diagonal. This helps to visually identify potential correlations between variables.
sns.pairplot(data)
plt.suptitle("Relationships between Simulated Variables", y=1.02)  # Adjust title position
plt.show()

# 6. Normalized Boxplots (for Visualization)
# To compare the spread and potential outliers of variables on a similar scale, the data is normalized using StandardScaler.
# StandardScaler scales the data so that it has a mean of 0 and a standard deviation of 1.
data_norm = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
plt.figure(figsize=(10, 4))
sns.boxplot(data=data_norm, palette="pastel")
plt.title("Normalized Boxplots of Simulated Variables")
plt.show()

# 7. Correlation Matrix
# The Pearson correlation coefficient is calculated for all pairs of variables in the original DataFrame.
# The correlation matrix shows the linear relationship between each pair of variables, ranging from -1 to 1.
correlation_matrix = data.corr()
print("\n🔹 Pearson correlation matrix:")
print(correlation_matrix.round(2))

# A heatmap is used to visualize the correlation matrix, where the color intensity and annotation indicate the strength and direction of the correlations.
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")  # 'annot=True' displays the correlation values, 'cmap' sets the color scheme
plt.title("Correlation Matrix between Variables")
plt.tight_layout()
plt.show()

# 8. Linear Regression: Cadence vs. Stride Length
# This section performs a linear regression analysis to model the relationship between 'Stride Length (m)' (independent variable) and 'Cadence (spm)' (dependent variable).
# scipy.stats.linregress calculates the slope, intercept, R-value, p-value, and standard error of the regression line.
x = data['Stride Length (m)']
y = data['Cadence (spm)']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"\n🔹 Linear regression: Cadence ~ Stride Length")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R²: {r_value**2:.2f}  (correlation = {r_value:.2f})")  # R-squared indicates the proportion of variance explained by the model
print(f"p-value: {p_value:.4f}")  # p-value indicates the significance of the relationship

# A scatter plot shows the original data points, and the fitted linear regression line is overlaid in red.
plt.figure(figsize=(6, 4))
sns.scatterplot(x=x, y=y)
plt.plot(x, intercept + slope * x, color='red', label=f'Linear regression\n$R^2$ = {r_value**2:.2f}')
plt.xlabel("Stride Length (m)")
plt.ylabel("Cadence (spm)")
plt.title("Relationship between Stride Length and Cadence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Correlation and Regression: Speed vs. Plantar Pressure
# This section examines the relationship between 'Speed (m/min)' and 'Plantar Pressure (kPa)'.
# First, the Pearson correlation coefficient between these two variables is calculated.
cor = data[['Speed (m/min)', 'Plantar Pressure (kPa)']].corr().iloc[0, 1]
print(f"\n🔹 Correlation between speed and plantar pressure: r = {cor:.2f}")

# A regression plot (using seaborn's regplot) visualizes the relationship, including a scatter plot of the data points and a linear regression line with a confidence interval.
plt.figure(figsize=(6, 4))
sns.regplot(x='Speed (m/min)', y='Plantar Pressure (kPa)', data=data,
            scatter_kws={"alpha":0.6}, line_kws={"color": "red"})
plt.title("Relationship between Speed and Plantar Pressure")
plt.xlabel("Speed (m/min)")
plt.ylabel("Plantar Pressure (kPa)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Data Standardization
# For clustering algorithms that are sensitive to the scale of the input features, the data is standardized using StandardScaler.
# Only the numerical features are selected for standardization.
features = ['Stride Length (m)', 'Cadence (spm)', 'Plantar Pressure (kPa)', 'Speed (m/min)']
X = StandardScaler().fit_transform(data[features])

# 11. Elbow Method for Choosing k (for K-Means)
# The elbow method is used to heuristically determine the optimal number of clusters (k) for K-Means.
# It involves running K-Means for a range of k values and plotting the within-cluster sum of squares (inertia).
# The "elbow" point in the plot (where the rate of decrease in inertia starts to slow down) is often considered a good estimate for k.
inertias = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10