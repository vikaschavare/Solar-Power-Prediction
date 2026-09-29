from flask import Flask, render_template, request
import pandas as pd
import joblib
import os


app = Flask(__name__)

MODEL_FILE = "solar_power_model.pkl"


# Load trained model
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "solar_power_model.pkl not found. "
        "Please run: python train_model.py"
    )

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
r2_score_value = model_data["r2_score"]
mae_value = model_data["mae"]


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        r2_score=r2_score_value,
        mae=mae_value
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        irradiance = float(request.form["irradiance"])
        wind_speed = float(request.form["wind_speed"])
        pressure = float(request.form["pressure"])

        # Create DataFrame
        input_data = pd.DataFrame({
            "temperature": [temperature],
            "humidity": [humidity],
            "irradiance": [irradiance],
            "wind_speed": [wind_speed],
            "pressure": [pressure]
        })

        # Predict solar power
        prediction = model.predict(input_data)[0]

        # Prevent negative value
        prediction = max(0, prediction)

        return render_template(
            "index.html",
            prediction=prediction,
            r2_score=r2_score_value,
            mae=mae_value,
            temperature=temperature,
            humidity=humidity,
            irradiance=irradiance,
            wind_speed=wind_speed,
            pressure=pressure
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=None,
            r2_score=r2_score_value,
            mae=mae_value,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)