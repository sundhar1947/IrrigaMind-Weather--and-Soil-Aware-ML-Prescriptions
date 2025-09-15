# file: rl_env.py
# Minimal RL environment for soil moisture control (OpenAI Gym-style)

import numpy as np

class SimpleIrrigationEnv:
    def __init__(self):
        # state: [soil_moisture, air_temp, humidity, et0, forecast_rain]
        self.state = None
        self.day = 0
        self.target_min = 30.0
        self.target_max = 60.0
        self.rng = np.random.default_rng(0)

        # action space: discrete mm: {0, 5, 10, 20}
        self.actions_mm = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float32)

    def reset(self):
        self.day = 0
        self.state = np.array([40.0, 28.0, 55.0, 4.0, 0.5], dtype=np.float32)
        return self.state.copy()

    def step(self, action_idx: int):
        soil, t, h, et0, rain = self.state
        irrig = self.actions_mm[action_idx]

        # soil balance: next_soil = soil + irrig + rain_gain - et0_loss + noise
        rain_gain = rain * 3.0  # mm to soil %
        et0_loss = et0 * 1.5
        noise = self.rng.normal(0, 1.0)
        next_soil = np.clip(soil + irrig + rain_gain - et0_loss + noise, 0, 100)

        # next weather
        t = np.clip(t + self.rng.normal(0, 1.0), 10, 40)
        h = np.clip(h + self.rng.normal(0, 3.0), 20, 90)
        et0 = np.clip(et0 + self.rng.normal(0, 0.5), 1, 8)
        rain = np.clip(max(0, rain + self.rng.normal(0, 0.2))), 0, 10)

        self.state = np.array([next_soil, t, h, et0, rain], dtype=np.float32)
        self.day += 1

        # reward: keep soil within band, penalize irrigation
        band_pen = 0.0
        if next_soil < self.target_min: band_pen -= (self.target_min - next_soil)
        if next_soil > self.target_max: band_pen -= (next_soil - self.target_max)
        water_pen = -0.2 * irrig
        reward = band_pen + water_pen
        done = self.day >= 120
        return self.state.copy(), reward, done, {}

# A lightweight DQN/Actor-Critic can be plugged here following cited repos/papers.
