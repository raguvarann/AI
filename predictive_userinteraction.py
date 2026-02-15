# predictive_rf.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 5, 8],
    'Price': [400000, 500000, 600000, 650000, 700000]
}
df = pd.DataFrame(data)

# Split features and target
X = df[['SquareFootage', 'Bedrooms', 'Age']]
y = df['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=30)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=30)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("Predicted Prices for Test Set:", y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}\n")

# Interactive user input for new prediction
try:
    square_footage = float(input("Enter square footage: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    age = int(input("Enter age of the house: "))

    new_house = pd.DataFrame([[square_footage, bedrooms, age]],
                             columns=['SquareFootage', 'Bedrooms', 'Age'])

    new_house_scaled = scaler.transform(new_house)
    predicted_price = model.predict(new_house_scaled)

    print(f"\nPredicted price for your house: ${predicted_price[0]:,.2f}")
except ValueError:
    print("Invalid input! Please enter numeric values for all fields.")
