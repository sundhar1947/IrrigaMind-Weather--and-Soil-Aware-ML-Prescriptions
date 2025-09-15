# file: serve_inference.py
# FastAPI inference service for real-time irrigation prescription

import json
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

FEATURES = [
    "soil_moisture","air_temp","air_humidity","wind_speed",
    "solar_rad","pressure","et0_mm","forecast_rain_mm","last_irrig_mm"
]

class InputPayload(BaseModel):
    soil_moisture: float
    air_temp: float
    air_humidity: float
    wind_speed: float
    solar_rad: float
    pressure: float
    et0_mm: float
    forecast_rain_mm: float
    last_irrig_mm: float

with open("scaler.json","r") as f:
    sc = json.load(f)
x_mean = np.array(sc["mean"], dtype=np.float32)
x_std  = np.array(sc["std"], dtype=np.float32)
model = torch.jit.load("predictor_best.ts", map_location="cpu")
model.eval()

app = FastAPI()

@app.post("/predict")
def predict(inp: InputPayload):
    x = np.array([getattr(inp, k) for k in FEATURES], dtype=np.float32)
    x = (x - x_mean) / (x_std + 1e-6)
    with torch.no_grad():
        y = model(torch.from_numpy(x).unsqueeze(0)).item()
    # clip to sensible range (0..40 mm daily typical), adjust to project context
    y = float(np.clip(y, 0.0, 40.0))
    return {"irrigation_mm": y}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
