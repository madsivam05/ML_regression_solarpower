# ML Regression — Solar Power Prediction

Predicts solar power output (watts) from weather conditions using a Linear Regression model.

**Inputs:** temperature (°C), humidity (%), solar irradiance (W/m²), wind speed (m/s)

## Files

| File | What it is |
|---|---|
| `solar_power.ipynb` | Notebook that trains and evaluates the model |
| `solar_power_prediction_model.pkl` | The trained model (saved with joblib) |
| `app.py` | Flask web app that serves the model at http://localhost:5000 |
| `templates/index.html` | Web page for the app |
| `requirements.txt` | All Python libraries needed |

## Setup (one time)

```bash
pip install -r requirements.txt
```

On Windows, if `pip` is not recognized, use:

```
py -m pip install -r requirements.txt
```

## Run the notebook

```
py -m jupyter notebook
```

Opens at http://localhost:8888 — click `solar_power.ipynb`.

## Run the web app

```
py app.py
```

Then open http://localhost:5000 in your browser, enter the weather values, and get a prediction.

## Quick prediction in Python

```python
import joblib
import pandas as pd

model = joblib.load("solar_power_prediction_model.pkl")
X = pd.DataFrame([[19.36, 75.85, 266.61, 5.19]],
                 columns=["temperature", "humidity", "solar_irradiance", "wind_speed"])
print(model.predict(X)[0])  # ~132.6 watts
```
