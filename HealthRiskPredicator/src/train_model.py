import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\Tanaya Tapas\Desktop\HealthRiskPredicator\Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv")

print("Columns:\n", df.columns)

# ✅ Correct target column
target_column = "health_risk"

# Separate input and output
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical columns
le_dict = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

# Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Save model
joblib.dump(model, "../model.pkl")

# Save encoders
joblib.dump(le_dict, "../encoders.pkl")
joblib.dump(target_encoder, "../target_encoder.pkl")

print("\nModel and encoders saved successfully!")