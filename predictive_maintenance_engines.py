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

# Engine usage and sensor data
data = {
    "Operating_Hours": np.random.uniform(10, 500, num_samples),  # Hours since last maintenance
    "Vibration_Level_g": np.random.uniform(0.1, 5.0, num_samples),  # Vibration in g-force
    "Temperature_degC": np.random.uniform(50, 200, num_samples),  # Engine temp
    "Pressure_bar": np.random.uniform(20, 100, num_samples),  # Internal pressure
    "Fuel_Flow_Rate_lps": np.random.uniform(5, 50, num_samples),  # Liters per second
}

# Combine into DataFrame
df = pd.DataFrame(data)

# Feature engineering
df["Vibration_Temp_Interaction"] = df["Vibration_Level_g"] * df["Temperature_degC"]

# Maintenance Need Score (0-1)
score = (
    0.4 * (df["Operating_Hours"] / 500) +  # More hours, higher need
    0.3 * (df["Vibration_Level_g"] / 5.0) +
    0.3 * (df["Temperature_degC"] - 50) / 150 +
    0.2 * (df["Pressure_bar"] - 20) / 80 -
    0.2 * (df["Fuel_Flow_Rate_lps"] / 50)  # Stable flow reduces need
)
df["Maintenance_Need_Score"] = np.clip(score, 0, 1)

# Define features (X) and target (y)
X = df.drop(columns=["Maintenance_Need_Score"])
y = df["Maintenance_Need_Score"]

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to 0-1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
timesteps = 1
features = X_train_scaled.shape[1]  # 6 features
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
    "Operating_Hours": {"max": 400},
    "Vibration_Level_g": {"max": 3.0},
    "Temperature_degC": {"max": 180},
    "Pressure_bar": {"max": 90},
    "Fuel_Flow_Rate_lps": {"min": 10}
}

# Prediction function for Gradio
def predict_maintenance_need(hours, vibration, temp, pressure, flow):
    input_data = np.array([[hours, vibration, temp, pressure, flow, vibration * temp]])
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape((1, timesteps, features))
    score = model.predict(input_scaled, verbose=0)[0][0]
    print(f"Raw score: {score}")

    # Check thresholds
    inputs = {
        "Operating_Hours": hours, "Vibration_Level_g": vibration,
        "Temperature_degC": temp, "Pressure_bar": pressure,
        "Fuel_Flow_Rate_lps": flow
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
    decision = "No Maintenance Needed" if score < threshold else "Maintenance Urgent"
    alert = "🚨 RED ALERT 🚨\n" + "\n".join(violations) if decision == "Maintenance Urgent" or violations else "No alert (Colab: Sound not supported)"

    return (f"Maintenance Need Score: {score:.4f}",
            f"Decision: {decision}",
            f"Threshold Violations: {', '.join(violations) if violations else 'None'}",
            alert)

# Gradio interface
interface = gr.Interface(
    fn=predict_maintenance_need,
    inputs=[
        gr.Slider(10, 500, label="Operating Hours", value=100),
        gr.Slider(0.1, 5.0, label="Vibration Level (g)", value=1.0),
        gr.Slider(50, 200, label="Temperature (°C)", value=100),
        gr.Slider(20, 100, label="Pressure (bar)", value=50),
        gr.Slider(5, 50, label="Fuel Flow Rate (l/s)", value=25),
    ],
    outputs=[
        gr.Textbox(label="Maintenance Score"),
        gr.Textbox(label="Maintenance Decision"),
        gr.Textbox(label="Threshold Violations"),
        gr.Textbox(label="Alert")
    ],
    title="Rocket Engine Maintenance Predictor",
    description="Enter engine usage and sensor data to predict maintenance needs. Red alert shows for urgent cases or violations (sound not supported in Colab)."
)

# Launch the interface
interface.launch()