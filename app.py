from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [
            data["age"],
            data["sex"],
            data["chol"],
            data["sysbp"],
            data["diabp"],
            data["bmi"],
            data["heartRate"],
            data["glucose"]
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
