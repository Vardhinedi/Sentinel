import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gradio as gr

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 samples
num_samples = 1000

# Weather conditions
weather_data = {
    "Temperature_C": np.random.uniform(-10, 40, num_samples),
    "Wind_Speed_kmph": np.random.uniform(0, 50, num_samples),
    "Atmospheric_Pressure_hPa": np.random.uniform(950, 1050, num_samples),
    "Humidity_percent": np.random.uniform(10, 100, num_samples),
    "Visibility_km": np.random.uniform(1, 20, num_samples),
    "Cloud_Cover_percent": np.random.uniform(0, 100, num_samples),
}

# Technical factors
technical_data = {
    "Engine_Thrust_kN": np.random.uniform(500, 1000, num_samples),
    "Fuel_Pump_Pressure_bar": np.random.uniform(50, 150, num_samples),
    "Avionics_Status": np.random.uniform(0.9, 1.0, num_samples),
    "Sensor_Reliability": np.random.uniform(0.85, 1.0, num_samples),
}

# Combine into DataFrame
df = pd.DataFrame({**weather_data, **technical_data})

# Feature engineering
df["Temp_Wind_Interaction"] = df["Temperature_C"] * df["Wind_Speed_kmph"]

# Launch Feasibility Score with strong penalties
weather_score = (
    0.4 * (df["Temperature_C"] / 40) -
    0.6 * (df["Wind_Speed_kmph"] / 50)**2 +
    0.3 * (df["Atmospheric_Pressure_hPa"] - 950) / 100 -
    0.5 * (df["Cloud_Cover_percent"] / 100)**2 -
    0.4 * (df["Humidity_percent"] / 100)
)
technical_score = (
    0.4 * (df["Engine_Thrust_kN"] - 500) / 500 +
    0.3 * (df["Fuel_Pump_Pressure_bar"] - 50) / 100 +
    0.2 * (df["Avionics_Status"] - 0.9) / 0.1 +
    0.1 * (df["Sensor_Reliability"] - 0.85) / 0.15
)
df["Launch_Feasibility_Score"] = 0.5 * weather_score + 0.5 * technical_score
df["Launch_Feasibility_Score"] = np.clip(df["Launch_Feasibility_Score"], 0, 1)

# Define features (X) and target (y)
X = df.drop(columns=["Launch_Feasibility_Score"])
y = df["Launch_Feasibility_Score"]

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to 0-1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
timesteps = 1
features = X_train_scaled.shape[1]  # 11 features
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, features))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, features))

# Define and train the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate on test set
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}")
print(f"R²: {r2}")
print(f"Prediction range: {y_pred.min():.4f} to {y_pred.max():.4f}, mean: {y_pred.mean():.4f}")

# Define thresholds
thresholds = {
    "Temperature_C": {"min": 0, "max": 35},
    "Wind_Speed_kmph": {"max": 30},
    "Atmospheric_Pressure_hPa": {"min": 970, "max": 1030},
    "Humidity_percent": {"max": 85},
    "Visibility_km": {"min": 5},
    "Cloud_Cover_percent": {"max": 70},
    "Engine_Thrust_kN": {"min": 600},
    "Fuel_Pump_Pressure_bar": {"min": 70},
    "Avionics_Status": {"min": 0.95},
    "Sensor_Reliability": {"min": 0.90}
}

# Prediction function for Gradio
def predict_launch_feasibility(temp, wind, pressure, humidity, visibility, clouds, thrust, pump_pressure, avionics, sensors):
    input_data = np.array([[temp, wind, pressure, humidity, visibility, clouds, 
                            thrust, pump_pressure, avionics, sensors, temp * wind]])
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape((1, timesteps, features))
    score = model.predict(input_scaled, verbose=0)[0][0]
    print(f"Raw score: {score}")  # Debug

    # Check thresholds
    inputs = {
        "Temperature_C": temp, "Wind_Speed_kmph": wind, "Atmospheric_Pressure_hPa": pressure,
        "Humidity_percent": humidity, "Visibility_km": visibility, "Cloud_Cover_percent": clouds,
        "Engine_Thrust_kN": thrust, "Fuel_Pump_Pressure_bar": pump_pressure,
        "Avionics_Status": avionics, "Sensor_Reliability": sensors
    }
    violations = []
    for feature, value in inputs.items():
        thresh = thresholds.get(feature, {})
        if "min" in thresh and value < thresh["min"]:
            violations.append(f"{feature}: {value} < {thresh['min']}")
        if "max" in thresh and value > thresh["max"]:
            violations.append(f"{feature}: {value} > {thresh['max']}")

    # Decision
    threshold = 0.7
    decision = "Good to go" if score >= threshold else "No go"
    alert = "🚨 RED ALERT 🚨\n" + "\n".join(violations) if decision == "No go" or violations else "No alert (Colab: Sound not supported)"

    return (f"Launch Feasibility Score: {score:.4f}",
            f"Decision: {decision}",
            f"Threshold Violations: {', '.join(violations) if violations else 'None'}",
            alert)

# Gradio interface
interface = gr.Interface(
    fn=predict_launch_feasibility,
    inputs=[
        gr.Slider(-10, 40, label="Temperature (°C)", value=25),
        gr.Slider(0, 50, label="Wind Speed (kmph)", value=10),
        gr.Slider(950, 1050, label="Atmospheric Pressure (hPa)", value=1010),
        gr.Slider(10, 100, label="Humidity (%)", value=50),
        gr.Slider(1, 20, label="Visibility (km)", value=15),
        gr.Slider(0, 100, label="Cloud Cover (%)", value=20),
        gr.Slider(500, 1000, label="Engine Thrust (kN)", value=750),
        gr.Slider(50, 150, label="Fuel Pump Pressure (bar)", value=100),
        gr.Slider(0.9, 1.0, label="Avionics Status (0.9-1.0)", value=0.95),
        gr.Slider(0.85, 1.0, label="Sensor Reliability (0.85-1.0)", value=0.95),
    ],
    outputs=[
        gr.Textbox(label="Feasibility Score"),
        gr.Textbox(label="Launch Decision"),
        gr.Textbox(label="Threshold Violations"),
        gr.Textbox(label="Alert")
    ],
    title="Rocket Launch Feasibility Predictor",
    description="Enter weather and technical data to predict if the rocket is good to launch. Red alert shows for 'No go' or violations (sound not supported in Colab)."
)

# Launch the interface
interface.launch()