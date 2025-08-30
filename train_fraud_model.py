# train_fraud_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # Better than pickle for saving models

# --- Generate a realistic synthetic dataset because we can't download the real one now ---
# This is for demonstration. In a real project, you would use the Kaggle dataset.
np.random.seed(42)  # For reproducible results
n_samples = 10000

# Simulate features: transaction amount, hour, and a derived feature (amount per hour)
data = {
    'amount': np.random.exponential(150, n_samples),  # Most transactions are small, few are large
    'hour': np.random.randint(0, 24, n_samples),      # Hour of the day
}
df = pd.DataFrame(data)
df['amount_to_hour_ratio'] = df['amount'] / (df['hour'] + 1) # Create a feature that might be predictive

# Simulate fraud labels: Fraud is more likely for large amounts at unusual hours
fraud_probability = 0.03 + (df['amount'] / 10000) + (np.abs(df['hour'] - 12) / 50)
df['is_fraud'] = np.random.binomial(1, fraud_probability) # 1 for fraud, 0 for genuine

# --- Prepare the data for the model ---
X = df[['amount', 'hour', 'amount_to_hour_ratio']] # Features the model learns from
y = df['is_fraud']                                 # Target variable we want to predict

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Create and Train the AI Model ---
print("Training the AI Fraud Detection Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Save the trained model to a file ---
model_filename = 'fraud_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Model trained and saved as {model_filename}")

# Optional: Check the model's performance
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
print("Sample predictions:", model.predict([[100, 14, 100/15], [5000, 3, 5000/4]])) # Predict on a small and large transaction
