from kafka import KafkaConsumer
import json
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# Load model
model = joblib.load("model.pkl")
le_dict = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

consumer = KafkaConsumer(
    'health_topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

conn = sqlite3.connect("health_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    user_id INTEGER,
    timestamp TEXT,
    age INTEGER,
    weight INTEGER,
    height INTEGER,
    exercise INTEGER,
    sleep INTEGER,
    sugar_intake INTEGER,
    smoking INTEGER,
    alcohol INTEGER,
    married INTEGER,
    profession TEXT,
    bmi REAL,
    predicted_risk TEXT
)
""")
conn.commit()

predictions_list = []

print("Kafka Consumer started... Listening to 'health_topic'")

for message in consumer:
    data = message.value
    df = pd.DataFrame([data])

    # 🔥 FIX HERE
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])

    # Encode
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Prediction
    prediction = model.predict(df)
    risk = target_encoder.inverse_transform(prediction)[0]

    data['predicted_risk'] = risk

    # Store in SQLite
    cursor.execute("""
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("user_id", 0),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("age"),
        data.get("weight"),
        data.get("height"),
        data.get("exercise"),
        data.get("sleep"),
        data.get("sugar_intake"),
        data.get("smoking"),
        data.get("alcohol"),
        data.get("married"),
        data.get("profession"),
        data.get("bmi"),
        risk
    ))

    conn.commit()

    predictions_list.append(data)

    print(f"Stored in SQLite: {data}")