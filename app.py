from flask import Flask, jsonify, request
import pandas as pd
import joblib
import requests
import json
import logging

app = Flask(__name__)

def send_data_to_laravel(data, endpoint, csrf_token):
    url = f'http://127.0.0.1:8000/api/{endpoint}'
    headers = {'Content-Type': 'application/json', 'X-CSRF-TOKEN': csrf_token}
    response = requests.post(url, json={'data': data}, headers=headers)
    return response

kmeans_model = joblib.load("data/kmeans_model.pkl")
scaler = joblib.load("data/scaler.pkl")

# Set up logging
logging.basicConfig(level=logging.DEBUG)

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
        x_scaled = scaler.transform(X)
        y_kmeans = kmeans_model.predict(x_scaled)
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
        logging.error(f"Failed to process file: {str(e)}")
        return f"Failed to process file: {str(e)}", 500

@app.route("/process_daftar", methods=["POST"])
def process_file_daftar():
    try:
        file = request.files["file"]
        df_daftar = pd.read_excel(file)

        df_clustered = pd.read_csv("data/clustered_data.csv")

        df_selection2 = pd.merge(
            df_daftar,
            df_clustered[["NPM", "Nama Mahasiswa", "Cluster", "Label Cluster"]],
            on="NPM",
            how="left",
            validate="one_to_one",
        )
        df_selection2.rename(columns={"Cluster": "Potensi"}, inplace=True)

        result = df_selection2.to_dict(orient="records")

        return jsonify(result)
    except Exception as e:
        logging.error(f"Failed to process file: {str(e)}")
        return f"Failed to process file: {str(e)}", 500

@app.route("/normalize", methods=["POST"])
def normalize_data():
    try:
        criteria = get_criteria()
        df = pd.read_csv(request.files.get("file"))

        # Normalisasi data berdasarkan kriteria
        for criteria_group, sub_criteria in criteria.items():
            for sub_criterion, details in sub_criteria.items():
                column_name = f"{criteria_group}_{sub_criterion}"
                if column_name in df.columns:
                    if details["type"] == "benefit":
                        df[column_name] = df[column_name] / df[column_name].max()
                    elif details["type"] == "cost":
                        df[column_name] = df[column_name].min() / df[column_name]

        # Kirim data ke Laravel untuk disimpan
        result = df.to_dict(orient="records")
        csrf_token = get_csrf_token()
        logging.debug(f"Sending data to Laravel with CSRF token: {csrf_token}")
        response = requests.post(
            "http://127.0.0.1:8000/store-normalized-data",
            headers={"Content-Type": "application/json", "X-CSRF-TOKEN": csrf_token},
            data=json.dumps(result),
        )
        logging.debug(f"Response from Laravel: {response.status_code}, {response.text}")
        if response.status_code != 200:
            raise Exception(
                f"Failed to store normalized data in Laravel: {response.text}"
            )

        return jsonify(result)
    except Exception as e:
        logging.error(f"Failed to normalize data: {str(e)}")
        return f"Failed to normalize data: {str(e)}", 500

@app.route("/saw", methods=["POST"])
def process_saw():
    try:
        criteria = get_criteria()
        logging.debug(f"Criteria: {criteria}")

        df = pd.read_csv(request.files.get("file"))
        logging.debug(f"Initial DataFrame: \n{df.head()}")

        for criteria_group, sub_criteria in criteria.items():
            for sub_criterion, details in sub_criteria.items():
                column_name = f"{criteria_group}_{sub_criterion}"
                if column_name in df.columns:
                    df[column_name] *= details["weight"]

        df["SAW_Score"] = df.sum(axis=1)
        df["Rank"] = df["SAW_Score"].rank(ascending=False)

        df.to_csv("data/saw_results.csv", index=False)
        result = df.to_dict(orient="records")
        logging.debug(f"SAW Results DataFrame: \n{df.head()}")

        # Dapatkan CSRF token
        csrf_token = get_csrf_token()

        # Kirim data ke Laravel untuk disimpan
        response = requests.post(
            "http://127.0.0.1:8000/store-saw-results",
            headers={"Content-Type": "application/json", "X-CSRF-TOKEN": csrf_token},
            data=json.dumps(result),
        )
        logging.debug(f"Response from Laravel: {response.status_code}, {response.text}")

        if response.status_code != 200:
            raise Exception("Failed to store SAW results in Laravel")

        return jsonify(result)
    except Exception as e:
        logging.error(f"Failed to process SAW: {str(e)}")
        return f"Failed to process SAW: {str(e)}", 500

def get_criteria():
    response = requests.get("http://127.0.0.1:8000/criteria")
    if response.status_code == 200:
        criteria_list = response.json()
        criteria = {}
        for item in criteria_list:
            criteria_group = item['criteria_group']
            sub_criteria = item['sub_criteria']
            if criteria_group not in criteria:
                criteria[criteria_group] = {}
            criteria[criteria_group][sub_criteria] = {
                'type': item['type'],
                'weight': float(item['weight'])
            }
        return criteria
    else:
        raise Exception("Failed to fetch criteria from Laravel")

def get_csrf_token():
    response = requests.get("http://127.0.0.1:8000/csrf-token")
    if response.status_code == 200:
        try:
            csrf_token = response.json()['csrf_token']
            logging.debug(f"CSRF Token: {csrf_token}")
            return csrf_token
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            logging.error("Failed to parse CSRF token from response")
            raise Exception("Failed to parse CSRF token from response") from e
    else:
        raise Exception("Failed to fetch CSRF token from Laravel")

def convert_grade_to_numeric(grade):
    grade_mapping = {"A": 90, "B": 80, "C": 70, "D": 60, "E": 50, "F": 40}
    return grade_mapping.get(grade, 0)

if __name__ == "__main__":
    app.run(debug=True)
