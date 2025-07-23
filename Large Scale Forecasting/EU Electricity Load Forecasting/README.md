# EU Electricity Load Forecasting

This project focuses on **forecasting the next 24 hours of electricity usage** for **24 European Union countries** using advanced **deep learning techniques**. It is part of a portfolio demonstrating best practices in large-scale time series modeling and scalable ML pipelines.

## 📈 Objective

> The goal is to build accurate and generalizable models that can forecast the hourly electricity load for each EU country over the next 24 hours, based on **past consumption**, **time-related features**, and **weather conditions**.

## 🔍 Problem Setup

- **Task:** Multivariate time series forecasting (hourly resolution)
- **Forecast Horizon:** 24 hours ahead
- **Granularity:** Country-level (24 EU countries)
- **Target Variable:** Electricity load (normalized)
- **Features Used:**
    - **Temporal features:** hour of day, day of week (encoded with sine/cosine)
    - **Calendar features:** `is_holiday`, `is_weekend`
    - **Weather features:**
        - `Temp_C` (Temperature)
        - `Humidity`
        - `WindSpeed`
        - `SolarRadiation`
        - `CloudCover`
- **Geographic context:** `CountryCode` (embedded via learned vector)

### Data Sources

This project combines two key datasets:

#### 1. Electricity Load Data
The electricity load (demand) data is sourced from the [ENTSO-E Transparency Platform](https://www.entsoe.eu/data/power-stats/). It provides:

- **Hourly load per country** (in GW or GWh).
- **Aggregated demand** across sectors — it does **not** separate residential, industrial, or commercial consumption.
- **Definition**: Load is defined as the **average instantaneous power demand** on the transmission/distribution grid for a given hour.

> Note: Sector-specific consumption breakdown is **not available**. Therefore, models forecast **total national load** rather than sector-specific demand.

---

#### 2. Weather Data
Hourly weather features were collected using the [Open-Meteo Archive API](https://open-meteo.com/), via a custom Python script that queries historical hourly data for each country using its central coordinates.

The weather variables include:

- `temperature_2m` → `Temp_C`  
- `relative_humidity_2m` → `Humidity`  
- `wind_speed_10m` → `WindSpeed`  
- `shortwave_radiation` → `SolarRadiation`  
- `cloud_cover` → `CloudCover`  

These variables were retrieved per country for the full range of dates covered in the electricity load dataset.

---
#### 🗓️ Date Range
The dataset covers the following period with **hourly resolution** (1-hour timestep):

- **Start**: `2023-01-01 00:00:00`  
- **End**: `2025-03-31 23:00:00`

---

#### 🗺️ Included EU Countries

Weather and electricity load data were successfully collected for the following countries:

`AT, BE, BG, CZ, DE, DK, EE, ES, FI, FR, GR, HR, HU, IT, LT, LU, LV, NL, PL, PT, RO, SE, SI`

Some EU countries were excluded due to one or more of the following reasons:
- Electricity load data was **not available** or incomplete on ENTSO-E.
- The country had **substantial missing values**

---


## 🧠 Approach

The pipeline is structured into clear stages:

Projects/
└── Time Series Forecasting/
    └── Large scale forecasting/
        ├── 1.EDA-Data Wrangling.ipynb
        ├── 2.Feature Engineering.ipynb
        ├── 3.Model Benchmarking.ipynb
        ├── environment.yml
        └── README.md  👈 (this file)


- **EDA & Wrangling:** Identifying seasonal patterns, missing values, country-level trends
- **Feature Engineering:** Encoding time features, country codes, scaling per country
- **Modeling:** Model Benchmarking
- **Evaluation:** MAE and RMSE reported per country and overall

## 🧠 Modeling Approach

Models implemented and compared include:
- Baseline
- Linear
- FNN
- LSTM
- CNN
- CNN-LSTM
- ARLSTM
- WaveNet

Each model was evaluated after tuning across multiple random seeds for fairness and generalization.

### Data Windowing Strategy

A custom `DataWindow` class is used to generate consistent sliding windows of historical input features and future target values. For each training sample:

- **Input window:** 24 hours of past data
- **Label window:** next 24 hours of target values (`Value`)

This setup aligns with the 24-hour ahead forecasting goal and allows batching across multiple countries with minimal leakage. The same structure is reused for training, validation, and test sets, and controls whether data is shuffled (e.g., during tuning) or not (e.g., during final evaluation).


## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- Boxplots and error distribution plots per country
- Best and worst prediction examples are highlighted

## ✅ Assumptions

- Although each country's electricity load is modeled independently, the deep learning model leverages shared temporal structures across countries—such as daily and weekly patterns—through a common architecture. While weather data is specific to each country, time-based patterns (like hour-of-day or day-of-week) and embeddings for country codes enable the model to generalize across similar temporal behaviors.
- Hourly patterns are cyclical and are encoded using sine and cosine transforms for features like hour of day and day of week.
- No external regressors (e.g., macroeconomic or industrial activity data) are used to maintain hourly resolution and reduce data sparsity.
- Holiday effects are assumed to be captured via a binary feature (is_holiday) at the country level.
- Weather data is assumed to be representative at the national level using a single central point (latitude/longitude) per country.
- Forecasting is performed using sliding windows (24 hours input → 24 hours output), and the model is trained to predict the next 24 hours for each window.
- All country models are trained jointly using country embeddings, but inference is evaluated per country.
- Target (Value) is assumed to be clean and continuous; extreme outliers (e.g., zero consumption) are not explicitly filtered out but implicitly handled by the model.

## 🛠️ Tools Used

- Python, Pandas, NumPy
- Tensorflow Forecasting
- Matplotlib, Seaborn
- Scikit-learn
- VS Code, GitHub


## 🚀 Future Work

- Add Data Center Consumption as regressor
- Experiment with probabilistic forecasts and uncertainty quantification
- Automate retraining and monitoring

---

**Author:** *Ehab Hasan*  
**Contact:** *https://www.linkedin.com/in/hasan87/*  
