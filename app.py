from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import datetime
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

users = {"admin": "password"}

# -------- Load Dataset and Preprocess -------- #
df = pd.read_csv("creditcard.csv")
categorical_cols = ['MerchantID', 'TransactionType', 'Location']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['TransactionDate'] = df['TransactionDate'].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)

df_sample = df.sample(n=5000, random_state=42)
X = df_sample[['Amount', 'MerchantID', 'TransactionType', 'Location', 'TransactionDate']]
y = df_sample['IsFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Global variables for models
models = {}
xgb_model = None
dnn_model = None
accuracies = {}

# -------- Load pre-trained model -------- #
dnn_model = load_model("dnn_model.h5")  # Load the pre-trained DNN model
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.load_model("xgb_model.json")  # Load the pre-trained XGBoost model

# -------- Lazy Training Functions -------- #
def train_classical_models():
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM": SVC(kernel='linear'),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    accs = {}
    trained = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accs[name] = round(accuracy_score(y_test, y_pred), 4)
        trained[name] = clf
    return trained, accs

# -------- ROUTES -------- #

@app.route('/')
def login_page():
    return render_template("auth.html")

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if username in users and users[username] == password:
        return jsonify({"message": "Login successful", "redirect": "/index"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if username in users:
        return jsonify({"error": "User already exists"}), 400
    users[username] = password
    return jsonify({"message": "Registration successful"})

@app.route('/index')
def index():
    global models, xgb_model, dnn_model, accuracies
    models, classical_acc = train_classical_models()

    # Load pre-trained models
    xgb_model, xgb_acc = train_xgb()
    dnn_model, dnn_acc = train_dnn()

    accuracies = classical_acc
    accuracies["XGBoost"] = xgb_acc
    accuracies["DNN"] = dnn_acc

    return render_template("index.html", accuracies=accuracies)

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    model_name = request.args.get("model_name")
    if model_name:
        normalized = model_name.replace("_", " ").replace("-", " ").lower()
        for key in accuracies:
            if key.lower() == normalized:
                return jsonify({key: accuracies[key]})
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if isinstance(data['TransactionDate'], str):
            try:
                data['TransactionDate'] = datetime.datetime.strptime(
                    data['TransactionDate'], "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            except Exception:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD HH:MM:SS"}), 400

        input_data = np.array([
            data['Amount'],
            data['MerchantID'],
            data['TransactionType'],
            data['Location'],
            data['TransactionDate']
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)
        model_type = data.get("model", "RF").upper()

        if model_type == "DL":
            prediction = dnn_model.predict(input_scaled)
            return jsonify({"prediction": int(prediction[0][0] > 0.5), "model": "DNN"})
        elif model_type == "XGB":
            prediction = xgb_model.predict(input_scaled)
            return jsonify({"prediction": int(prediction[0]), "model": "XGBoost"})
        else:
            prediction = models["Random Forest"].predict(input_scaled)
            return jsonify({"prediction": int(prediction[0]), "model": "Random Forest"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- MAIN -------- #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
