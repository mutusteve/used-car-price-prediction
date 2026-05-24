
# Model Card — Used Car Price Predictor

## Model Details
- **Name:** Ridge Regression
- **Type:** Ridge
- **Version:** 1.0.0
- **Created:** 20260524_201309
- **Author:** Stephen Mutu 

## Intended Use
- Predict the fair market selling price of used cars in India.

## Training Data
- Source: CarDekho Vehicle Dataset (Kaggle)
- Size: 301 rows
- Features: 8 engineered features

## Performance

| Metric | Value |
|--------|-------|
| Test R² | 0.9443 |
| Test RMSE | ₹1.7107 Lakhs |
| Test MAE | ₹0.9212 Lakhs |

## Features Used

- Present_Price_log
- Car_Age
- Kms_Driven_log
- Fuel_Diesel
- Fuel_Petrol
- Transmission_enc
- Seller_Type_enc
- Owner

## Preprocessing Steps
1. Log-transform numerical columns
2. Feature engineering
3. Encoding categorical variables
4. Scaling selected features

## Limitations
- Small dataset
- India-only market
- No accident-history data

## Ethical Considerations
- Predictions are estimates only
- Users should compare multiple valuations

## Example Usage

    import pickle
    import json

    with open('models/champion_model.pkl', 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully")
