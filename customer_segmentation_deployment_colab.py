
# =====================================================
# Customer Segmentation Deployment Script (Colab-ready)
# =====================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import pickle
import joblib
import shutil

from google.colab import drive
drive.mount('/content/drive')

# Step 2: Import libraries and load dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import pickle
import joblib
import shutil

# Step 3: Load your cleaned dataset (update the path if needed)
# Example path: /content/drive/MyDrive/customer_segmentation/cleaned_customer_data.csv
df = pd.read_csv('/content/drive/MyDrive/customer_segmentation/cleaned_customer_data.csv')

# Step 4: Feature selection based on your notebook
features = ['Income', 'Recency', 'TotalSpend', 'Age', 'Customer_For', 'TotalKids', 'NumWebVisitsMonth']
X = df[features].copy()

# Step 5: Preprocessing and model training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Auto-select best number of clusters
best_k, best_score = 0, -1
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}, silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score = k, score

print(f"Best k={best_k}, silhouette={best_score:.4f}")

# Fit final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
kmeans.fit(X_pca)

# Step 6: Create pipeline and save model, scaler, PCA
pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('kmeans', kmeans)
])

with open('/content/kmeans_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Also copy to Google Drive
drive_path = '/content/drive/MyDrive/customer_segmentation'
shutil.copy('/content/kmeans_pipeline.pkl', f'{drive_path}/kmeans_pipeline.pkl')

print("✅ Model pipeline saved successfully at:", drive_path)

# Step 7: Load and predict (inference)
with open(f'{drive_path}/kmeans_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input for a new customer
new_customer = {
    'Income': 0,     'Recency': 0,     'TotalSpend': 0,     'Age': 0,     'Customer_For': 0,     'TotalKids': 0,     'NumWebVisitsMonth': 0
}

X_new = pd.DataFrame([new_customer])
prediction = model.predict(X_new)
print("Predicted Cluster for new customer:", int(prediction[0]))
