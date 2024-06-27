from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan scaler yang sudah dilatih
kmeans_model = joblib.load("data/kmeans_model.pkl")
scaler = joblib.load("data/scaler.pkl")


@app.route("/")
def index():
    return "Flask API is running."


@app.route("/process", methods=["POST"])
def process_file():
    try:
        file = request.files["file"]
        df = pd.read_excel(file)

        df["Algoritma_Numeric"] = df["Algoritma"].apply(convert_grade_to_numeric)
        df["Statistika_Numeric"] = df["Statistika"].apply(convert_grade_to_numeric)

        def determine_peminatan(row):
            if row["Algoritma_Numeric"] > row["Statistika_Numeric"]:
                return "Kodingan"
            elif row["Statistika_Numeric"] > row["Algoritma_Numeric"]:
                return "Data"
            elif (
                row["Algoritma_Numeric"] <= 80
                and row["Statistika_Numeric"] <= 80
                and row["Nilai Project"] < 80
            ):
                return "Tidak Diketahui"
            else:
                return "Tidak Diketahui"

        df["Peminatan"] = df.apply(determine_peminatan, axis=1)

        features = [
            "Algoritma_Numeric",
            "Statistika_Numeric",
            "Nilai Project",
            "Kedisiplinan Akademik",
            "Keaktifan",
        ]
        X = df[features]
        X_scaled = scaler.transform(X)
        y_kmeans = kmeans_model.predict(X_scaled)
        label_mapping = {1: 0, 0: 3, 2: 2, 3: 1}
        y_kmeans_mapped = pd.Series(y_kmeans).map(label_mapping).values

        df["Cluster"] = y_kmeans_mapped
        cluster_labels = {
            0: "Potensi Tinggi",
            1: "Potensi Rendah",
            2: "Tidak ada potensi",
            3: "Potensi Sedang",
        }

        df["Label Cluster"] = df["Cluster"].map(cluster_labels)
        df.to_csv("data/clustered_data.csv", index=False)
        result = df.to_dict(orient="records")

        return jsonify(result)
    except Exception as e:
        return f"Failed to process file: {str(e)}", 500


@app.route('/process_daftar', methods=['POST'])
def process_file_daftar():
    try:
        file = request.files['file']
        df_daftar = pd.read_excel(file)

        # Ambil data mahasiswa yang sudah diklasterisasi
        df_clustered = pd.read_csv('data/clustered_data.csv')

        # Gabungkan data pendaftar dengan data mahasiswa
        df_selection2 = pd.merge(
            df_daftar,
            df_clustered[["NPM", "Nama Mahasiswa", "Cluster", "Label Cluster"]],
            on="NPM",
            how="left",
            validate="one_to_one"
        )
        df_selection2.rename(columns={"Cluster": "Potensi"}, inplace=True)

        # Konversi ke format JSON untuk dikirimkan ke Laravel
        result = df_selection2.to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return f"Failed to process file: {str(e)}", 500


def convert_grade_to_numeric(grade):
    grade_mapping = {"A": 90, "B": 80, "C": 70, "D": 60, "E": 50, "F": 40}
    return grade_mapping.get(grade, 0)


if __name__ == "__main__":
    app.run(debug=True)
