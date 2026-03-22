#Virtual Guardian Backend 

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from collections import deque
from datetime import datetime
import requests
from sklearn.ensemble import IsolationForest
import os

app = FastAPI(title="Vital Guardian Advanced Monitoring API")

#Loading Model 

model = joblib.load("overdose_monitor_model.pkl")

#Isolation Forest for anomaly detection
anomaly_model = IsolationForest(contamination=0.02)

#Configuration 

RED_THRESHOLD = 0.50
ORANGE_THRESHOLD = 0.50
SUSTAIN_TIME = 20

SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"

TWILIO_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH = "YOUR_TWILIO_AUTH"
TWILIO_PHONE = "YOUR_TWILIO_NUMBER"

#Memory Buffer 

user_buffers = {}

#Input Schema 

class VitalInput(BaseModel):
    user_id: str
    heart_rate: float
    resp_rate: float
    spo2: float
    movement_index: float
    rr_avg: float
    spo2_avg: float
    hr_avg: float

#Database helpers 

def log_to_supabase(data):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    requests.post(
        f"{SUPABASE_URL}/rest/v1/vitals_log",
        headers=headers,
        json=data
    )

def get_patient_phone(user_id):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/patients?user_id=eq.{user_id}",
        headers=headers
    )

    data = r.json()
    if data:
        return data[0]["emergency_phone"]
    return None

#SMS ALERT

def send_sms_alert(phone, message):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"

    requests.post(
        url,
        data={
            "From": TWILIO_PHONE,
            "To": phone,
            "Body": message
        },
        auth=(TWILIO_SID, TWILIO_AUTH)
    )

#Sustained RED logic

def apply_sustained_logic(user_id, predicted_stage):

    if user_id not in user_buffers:
        user_buffers[user_id] = deque(maxlen=SUSTAIN_TIME)

    buffer = user_buffers[user_id]
    buffer.append(predicted_stage)

    if len(buffer) == SUSTAIN_TIME and all(stage == 2 for stage in buffer):
        return 2

    if predicted_stage == 2:
        return 1

    return predicted_stage

#Prediction endpoint

@app.post("/predict")
def predict(vitals: VitalInput):

    features = np.array([[
        vitals.heart_rate,
        vitals.resp_rate,
        vitals.spo2,
        vitals.movement_index,
        vitals.rr_avg,
        vitals.spo2_avg,
        vitals.hr_avg
    ]])

    #ML prediction
    probs = model.predict_proba(features)[0]

    if probs[2] > RED_THRESHOLD:
        raw_stage = 2
    elif probs[1] > ORANGE_THRESHOLD:
        raw_stage = 1
    else:
        raw_stage = 0

    #Anomaly detection
    anomaly_score = anomaly_model.fit_predict(features)[0]
    if anomaly_score == -1:
        raw_stage = max(raw_stage, 1)

    final_stage = apply_sustained_logic(vitals.user_id, raw_stage)

    #Log to database
    log_to_supabase({
        "user_id": vitals.user_id,
        "heart_rate": vitals.heart_rate,
        "resp_rate": vitals.resp_rate,
        "spo2": vitals.spo2,
        "movement_index": vitals.movement_index,
        "stage": final_stage,
        "timestamp": datetime.utcnow().isoformat()
    })

    #Trigger SMS if RED
    if final_stage == 2:
        phone = get_patient_phone(vitals.user_id)
        if phone:
            send_sms_alert(
                phone,
                f"🚨 ALERT: Possible overdose detected for {vitals.user_id}. Immediate attention required."
            )

    return {
        "green_prob": float(probs[0]),
        "orange_prob": float(probs[1]),
        "red_prob": float(probs[2]),
        "stage": int(final_stage)
    }
