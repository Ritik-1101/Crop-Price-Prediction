import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data_path = 'ProductPriceIndex.csv'

data = pd.read_csv(data_path)

# Convert price fields from strings to numeric values and handle missing values
data['farmprice'] = pd.to_numeric(data['farmprice'].str.replace('$', ''), errors='coerce')
data.dropna(subset=['farmprice'], inplace=True)

# Convert 'date' to datetime format and extract 'year' and 'month'
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Define features and target
features = ['productname', 'year', 'month']
X = data[features]
y = data['farmprice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for categorical features
categorical_features = ['productname']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')

# Define the Random Forest model pipeline
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', rf_model)])

# Train the model
pipeline.fit(X_train, y_train)

# Model evaluation
y_pred = pipeline.predict(X_test)
print(f"Model RMSE on test set: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Function to make predictions with handling unknown categories
def predict_farm_price(productname, year, month, model=pipeline):
    input_df = pd.DataFrame([[productname, year, month]], columns=['productname', 'year', 'month'])
    predicted_price = model.predict(input_df)[0]
    return predicted_price

# Interactive user input for prediction
product_name = input("Enter product name: ")
year = int(input("Enter year (e.g., 2021): "))
month = int(input("Enter month (1-12): "))

predicted_price = predict_farm_price(product_name, year, month)
print(f"Predicted farm price for {product_name} in {month}/{year}: ${predicted_price:.2f}")

