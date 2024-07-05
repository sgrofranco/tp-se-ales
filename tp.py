import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ruta al archivo CSV descargado (ajusta según tu ubicación de descarga)
file_path = 'C:/repositorios vc/SEÑALES/TP FINAL/Emotions.csv'  # Ejemplo: 'path/to/emotions.csv'

# Cargar los datos del archivo CSV
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
print(data.head())

# Asumimos que las etiquetas están en una columna llamada 'label' y las características en otras columnas
X = data.drop(columns=['label'])
y = data['label']

# Convertir etiquetas categóricas a numéricas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocesar datos: Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducir dimensionalidad con PCA
n_components = 50  # Probar con 50 componentes principales
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Entrenar una SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM con {n_components} componentes PCA: {accuracy * 100:.2f}%")

# Reducir dimensionalidad con t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X_scaled)

# Visualizar los resultados de t-SNE
plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='viridis')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.title('Visualización de t-SNE de los datos EEG')
plt.show()

# Ver las clases codificadas
print("Clases codificadas por LabelEncoder:", label_encoder.classes_)

