from kafka import KafkaProducer
import json
import time
import random

# Create producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Sample categories
professions = ["student", "office", "self-employed"]

def generate_data():
    age = random.randint(20, 60)
    weight = random.randint(50, 100)
    height = random.randint(150, 180)
    bmi = round(weight / ((height / 100) ** 2), 2)

    data = {
        "user_id": random.randint(1, 5),  # IMPORTANT for Redis key
        "age": age,
        "weight": weight,
        "height": height,
        "exercise": random.randint(0, 5),
        "sleep": random.randint(4, 9),
        "sugar_intake": random.randint(1, 5),
        "smoking": random.randint(0, 1),
        "alcohol": random.randint(0, 3),
        "married": random.randint(0, 1),
        "profession": random.choice(professions),
        "bmi": bmi
    }

    return data


# Continuous streaming
while True:
    data = generate_data()

    producer.send('health_topic', value=data)
    print("Sent:", data)

    time.sleep(3)

# Flush (not really reached due to loop, but safe)
producer.flush()