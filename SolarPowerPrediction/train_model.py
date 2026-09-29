import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


CSV_FILE = "solar_power.csv"
MODEL_FILE = "solar_power_model.pkl"

FEATURES = [
    "temperature",
    "humidity",
    "irradiance",
    "wind_speed",
    "pressure"
]

TARGET = "solar_power_kw"


def create_dataset():
    """Create a realistic demo solar power dataset."""

    np.random.seed(42)

    n = 2000

    temperature = np.random.uniform(10, 45, n)
    humidity = np.random.uniform(20, 95, n)
    irradiance = np.random.uniform(50, 1100, n)
    wind_speed = np.random.uniform(0, 15, n)
    pressure = np.random.uniform(990, 1030, n)

    # Generate solar power values
    temperature_effect = 1 - 0.003 * np.maximum(temperature - 25, 0)
    humidity_effect = 1 - 0.0015 * np.maximum(humidity - 60, 0)
    pressure_effect = 1 + 0.0002 * (pressure - 1013)

    solar_power = (
        irradiance
        * 0.008
        * temperature_effect
        * humidity_effect
        * pressure_effect
    )

    # Add small random noise
    noise = np.random.normal(0, 0.15, n)

    solar_power = solar_power + noise

    # Power cannot be negative
    solar_power = np.maximum(solar_power, 0)

    data = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "irradiance": irradiance,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "solar_power_kw": solar_power
    })

    data.to_csv(CSV_FILE, index=False)

    print("New dataset created successfully.")
    print(f"Dataset saved as: {CSV_FILE}")


def dataset_is_valid():
    """Check whether the existing CSV is usable."""

    if not os.path.exists(CSV_FILE):
        return False

    try:
        data = pd.read_csv(CSV_FILE)

        required_columns = FEATURES + [TARGET]

        if len(data) < 50:
            return False

        if not all(column in data.columns for column in required_columns):
            return False

        if data[TARGET].nunique() < 2:
            return False

        return True

    except Exception:
        return False


# Create a valid dataset if required
if not dataset_is_valid():
    create_dataset()


# Load dataset
data = pd.read_csv(CSV_FILE)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# Separate input and output
X = data[FEATURES]
y = data[TARGET]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=15
)


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print("\nModel Evaluation")
print("----------------")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f} kW")


# Save model and evaluation values
model_data = {
    "model": model,
    "r2_score": r2,
    "mae": mae,
    "features": FEATURES
}

joblib.dump(model_data, MODEL_FILE)

print("\nModel trained successfully!")
print(f"Model saved as: {MODEL_FILE}")