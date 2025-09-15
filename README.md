# IrrigaMind-Weather--and-Soil-Aware-ML-Prescriptions

Predicts exact irrigation needs from real-time soil and weather data using a PyTorch model, deployed through a LoRa-based IoT stack for field sensing and automated watering. Designed for sustainable agriculture and real-world impact, with a clear upgrade path to reinforcement learning.

---

## 🚀 Highlights
- Predicts **daily irrigation in mm** from soil moisture, on-site weather, ET0, and forecast rain using a compact PyTorch model.  
- Uses **Arduino/ESP32 LoRa nodes** for low-power, long-range soil sensing and an ESP32 gateway to actuate pumps/valves.  
- Production-friendly: **TorchScript inference service**, simple HTTP API, and safe actuation logic.  
- **RL-ready**: Swap regression for a learned policy to minimize water use while keeping moisture in target bands.  

---

## 📂 Repository structure
- `train_predictor.py` — Train supervised PyTorch model to predict `irrigation_mm`.  
- `serve_inference.py` — FastAPI microservice that normalizes inputs and returns predictions from TorchScript model.  
- `lora_node_transmitter.ino` — Arduino LoRa transmitter: soil moisture + DHT temp/humidity → JSON payload.  
- `lora_gateway_esp32.ino` — ESP32 LoRa gateway: augments data, calls inference API, controls pump.  
- `rl_env.py` — Minimal Gym-style irrigation environment for RL (DQN/actor-critic).  
- `data/irrigation_samples.csv` — Example training data schema.  

---

## 🌱 Problem and approach
Over/under-irrigation wastes water and reduces yield. Manual schedules ignore soil-water dynamics.  
This project predicts **daily irrigation requirement in mm** using supervised ML on soil moisture, microclimate, ET0, and forecast rain.  
LoRa telemetry + inference service → pump runtime conversion → **closed-loop irrigation**.  

---

## 📊 Data schema
**One row per day per zone/plot**  

Inputs:  
- `soil_moisture, air_temp, air_humidity, wind_speed, solar_rad, pressure, et0_mm, forecast_rain_mm, last_irrig_mm`  
Target:  
- `target_mm` (required irrigation that day)  

Example columns:  
`timestamp, soil_moisture, air_temp, air_humidity, wind_speed, solar_rad, pressure, et0_mm, forecast_rain_mm, last_irrig_mm, target_mm`

---

## 🧠 Model training
- Baseline: **MLP + dropout, Adam optimizer, ReduceLROnPlateau scheduler**.  
- Normalization: saves mean/std → `scaler.json`.  
- Best weights → `predictor_best.pt` → TorchScript `predictor_best.ts`.  
- Outputs clipped to **0–40 mm/day**.  

Run:
```bash
python train_predictor.py
