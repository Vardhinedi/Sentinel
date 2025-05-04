
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

# Orbital and operational data
data = {
    "Orbit_Altitude_km": np.random.uniform(400, 800, num_samples),  # LEO range
    "Inclination_deg": np.random.uniform(0, 90, num_samples),  # Orbital tilt
    "Ground_Station_Availability": np.random.uniform(0.7, 1.0, num_samples),  # % available
    "Power_Level_percent": np.random.uniform(60, 100, num_samples),  # Battery/solar
    "Data_Storage_percent": np.random.uniform(20, 90, num_samples),  # Storage used
    "Target_Priority": np.random.uniform(0.5, 1.0, num_samples),  # Mission importance
}

# Combine into DataFrame
df = pd.DataFrame(data)

# Feature engineering
df["Altitude_Power_Interaction"] = df["Orbit_Altitude_km"] * df["Power_Level_percent"]

# Task Priority Score (0-1)
score = (
    0.3 * (df["Orbit_Altitude_km"] - 400) / 400 +  # Higher altitude better
    0.2 * (df["Inclination_deg"] / 90) +
    0.3 * (df["Ground_Station_Availability"] - 0.7) / 0.3 +
    0.2 * (df["Power_Level_percent"] - 60) / 40 -
    0.4 * (df["Data_Storage_percent"] / 100) +  # Penalize high storage
    0.3 * (df["Target_Priority"] - 0.5) / 0.5
)
df["Task_Priority_Score"] = np.clip(score, 0, 1)

# Define features (X) and target (y)
X = df.drop(columns=["Task_Priority_Score"])
y = df["Task_Priority_Score"]

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features to 0-1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
timesteps = 1
features = X_train_scaled.shape[1]  # 7 features
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
    "Orbit_Altitude_km": {"min": 450},
    "Inclination_deg": {"min": 30},
    "Ground_Station_Availability": {"min": 0.8},
    "Power_Level_percent": {"min": 70},
    "Data_Storage_percent": {"max": 80},
    "Target_Priority": {"min": 0.6}
}

# Prediction function for Gradio
def predict_task_priority(altitude, inclination, ground_station, power, storage, priority):
    input_data = np.array([[altitude, inclination, ground_station, power, storage, priority, 
                            altitude * power]])
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape((1, timesteps, features))
    score = model.predict(input_scaled, verbose=0)[0][0]
    print(f"Raw score: {score}")

    # Check thresholds
    inputs = {
        "Orbit_Altitude_km": altitude, "Inclination_deg": inclination,
        "Ground_Station_Availability": ground_station, "Power_Level_percent": power,
        "Data_Storage_percent": storage, "Target_Priority": priority
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
    decision = "High Priority" if score >= threshold else "Low Priority"
    alert = "🚨 RED ALERT 🚨\n" + "\n".join(violations) if decision == "Low Priority" or violations else "No alert (Colab: Sound not supported)"

    return (f"Task Priority Score: {score:.4f}",
            f"Decision: {decision}",
            f"Threshold Violations: {', '.join(violations) if violations else 'None'}",
            alert)

# Gradio interface
interface = gr.Interface(
    fn=predict_task_priority,
    inputs=[
        gr.Slider(400, 800, label="Orbit Altitude (km)", value=600),
        gr.Slider(0, 90, label="Inclination (deg)", value=45),
        gr.Slider(0.7, 1.0, label="Ground Station Availability (0.7-1.0)", value=0.9),
        gr.Slider(60, 100, label="Power Level (%)", value=85),
        gr.Slider(20, 90, label="Data Storage Used (%)", value=50),
        gr.Slider(0.5, 1.0, label="Target Priority (0.5-1.0)", value=0.8),
    ],
    outputs=[
        gr.Textbox(label="Priority Score"),
        gr.Textbox(label="Task Decision"),
        gr.Textbox(label="Threshold Violations"),
        gr.Textbox(label="Alert")
    ],
    title="Satellite Tasking Optimizer",
    description="Enter orbital and operational data to prioritize satellite tasks. Red alert shows for low priority or violations (sound not supported in Colab)."
)

# Launch the interface
interface.launch()