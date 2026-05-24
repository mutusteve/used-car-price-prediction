# Used Car Price Prediction 🚗

A complete end-to-end machine learning project that predicts 
the selling price of used cars using the CarDekho dataset.

---

## Project Overview

Used car pricing is complex — it depends on age, mileage, 
fuel type, transmission, and more. This project builds a 
regression model that predicts a car's selling price in 
Indian Rupees (Lakhs) given its key attributes.

**Dataset:** [Vehicle Dataset from CarDekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)  
**Records:** 301 used car listings  
**Target:** Selling Price (Lakhs ₹)

---

## Key Results

| Model | R² Score | RMSE (Lakhs) | MAE (Lakhs) |
|---|---|---|---|
| **Linear Regression** | **0.9745** | **₹1.71** | **₹0.92** |
| XGBoost | 0.9424 | ₹4.01 | ₹1.22 |
| Random Forest | 0.9067 | ₹4.70 | ₹1.73 |


---

````markdown
# Project Structure

```text
used_car_price_prediction/
│
├── data/
│   ├── raw/                         ← original dataset (not tracked by git)
│   └── processed/                   ← cleaned and engineered features
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_and_visualisation.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_feature_selection_and_split.ipynb
│
├── models/                          ← saved model files (.pkl)
│
├── app/                             ← Streamlit prediction app
│
├── requirements.txt                 ← Python dependencies
│
└── README.md
````

```
```


## Workflow

1. **Data Cleaning** — handle missing values, fix types, remove outliers, engineer `Car_Age`
2. **EDA** — distribution analysis, correlation heatmap, categorical vs price plots
3. **Feature Engineering** — log transforms, label encoding, one-hot encoding
4. **Feature Selection** — correlation, Random Forest importance, SelectKBest
5. **Modelling** — Linear Regression, Random Forest, XGBoost *(in progress)*
6. **Evaluation** — RMSE, MAE, R² on held-out test set
7. **Deployment** — Streamlit web app for live predictions

---

## Key Findings from EDA

- `Present_Price` is the strongest predictor of selling price
- Older cars depreciate significantly — `Car_Age` has a strong negative correlation
- Diesel cars command a price premium over Petrol
- Automatic transmission cars sell for more than Manual
- Cars with 0 previous owners have the highest resale value
- `Selling_Price` is right-skewed — log transformation applied before modelling

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Visualisation |
| scikit-learn | ML models, preprocessing |
| XGBoost | Gradient boosting model |
| pickle | Model serialisation |
| Streamlit | Prediction web app |
| Jupyter Notebook | Development environment |

---

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/used_car_price_prediction.git
cd used_car_price_prediction

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat        # Windows
# source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Place 'car data.csv' into data/raw/

# 5. Run notebooks in order (01 → 05)
jupyter notebook
```

---

## Author

---

## License

This project is open source under the [MIT License](LICENSE).
