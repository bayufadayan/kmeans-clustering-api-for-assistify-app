import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Fungsi untuk mengkonversi nilai huruf ke numerik
def convert_grade_to_numeric(grade):
    grade_mapping = {
        'A': 90,
        'B': 80,
        'C': 70,
        'D': 60,
        'E': 50,
        'F': 40
    }
    return grade_mapping.get(grade, 0)

# Load data dari file Excel
df = pd.read_excel('data/spk_mhs_data_fix.xlsx')

# Tambahkan kolom numerik untuk nilai huruf
df['Algoritma_Numeric'] = df['Algoritma'].apply(convert_grade_to_numeric)
df['Statistika_Numeric'] = df['Statistika'].apply(convert_grade_to_numeric)

# Tentukan peminatan berdasarkan nilai numerik
def determine_peminatan(row):
    if row['Algoritma_Numeric'] > row['Statistika_Numeric']:
        return 'Kodingan'
    elif row['Statistika_Numeric'] > row['Algoritma_Numeric']:
        return 'Data'
    elif row['Algoritma_Numeric'] <= 80 and row['Statistika_Numeric'] <= 80 and row['Nilai Project'] < 80:
        return 'Tidak Diketahui'
    else:
        return 'Tidak Diketahui'

df['Peminatan'] = df.apply(determine_peminatan, axis=1)

# Pilih fitur untuk clustering
features = ['Algoritma_Numeric', 'Statistika_Numeric', 'Nilai Project', 'Kedisiplinan Akademik', 'Keaktifan']
X = df[features]

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster optimal
optimal_clusters = 4

# Latih model KMeans
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X_scaled)

# Tambahkan hasil clustering ke dataframe
df['Cluster'] = y_kmeans

# Label clustering
cluster_labels = {
    0: 'Potensi Sedang',
    1: 'Potensi Tinggi',
    2: 'Tidak ada potensi',
    3: 'Potensi Rendah'
}

df['Label Cluster'] = df['Cluster'].map(cluster_labels)

# Simpan dataframe dengan hasil clustering ke file CSV
df.to_csv('data/clustered_data.csv', index=False)

# Simpan model dan skaler
joblib.dump(kmeans, 'data/kmeans_model.pkl')
joblib.dump(scaler, 'data/scaler.pkl')

print("Model, skaler, dan hasil clustering berhasil disimpan.")