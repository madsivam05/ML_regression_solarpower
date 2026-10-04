"""
Local web server for the Solar Power Prediction model.

Run it with:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained LinearRegression model (saved with joblib in the notebook)
model = joblib.load("solar_power_prediction_model.pkl")

FEATURES = ["temperature", "humidity", "solar_irradiance", "wind_speed"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    values = {f: "" for f in FEATURES}

    if request.method == "POST":
        try:
            values = {f: request.form.get(f, "") for f in FEATURES}
            x = np.array([[float(values[f]) for f in FEATURES]])
            prediction = round(float(model.predict(x)[0]), 2)
        except ValueError:
            prediction = "error"

    return render_template("index.html", prediction=prediction, values=values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST {"temperature": 30, "humidity": 40, "solar_irradiance": 800, "wind_speed": 5}"""
    data = request.get_json(force=True)
    try:
        x = np.array([[float(data[f]) for f in FEATURES]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}. Required fields: {FEATURES}"}), 400
    return jsonify({"solar_power_output": float(model.predict(x)[0])})


if __name__ == "__main__":
    # host="0.0.0.0" makes it reachable on your network too;
    # use host="127.0.0.1" if you want it strictly local-only.
    app.run(host="0.0.0.0", port=5000, debug=False)
