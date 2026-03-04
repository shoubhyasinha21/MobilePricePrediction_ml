import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load your CSV
df = pd.read_csv("C:\\Users\\1099TU\\OneDrive\\Documents\\mobile_phone_pricing_ml\\mobile_dataset.csv")

# Define exactly the 6 features you want in your app
features = ['battery_power', 'ram', 'px_height', 'px_width', 'mobile_wt', 'int_memory']
X = df[features]
y = df['price_range']

# Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save the 6-feature versions
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")