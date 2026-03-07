import pandas as pd
import numpy as np
import joblib
import os  # Added to handle directory creation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Create the 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# 2. Load your CSV
# Tip: Using a relative path like "mobile_dataset.csv" works if the file is in the same folder as this script
df = pd.read_csv("C:\\Users\\1099TU\\OneDrive\\Documents\\mobile_phone_pricing_ml\\mobile_dataset.csv")

# 3. Define exactly the 6 features you want in your app
features = ['battery_power', 'ram', 'px_height', 'px_width', 'mobile_wt', 'int_memory']
X = df[features]
y = df['price_range']

# 4. Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# 5. Save the 6-feature versions
# This will now work because we created the 'model' folder in step 1
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and Scaler saved successfully in the 'model/' folder!")