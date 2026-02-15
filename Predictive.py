import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 5, 8],
    'Price': [400000, 500000, 600000, 650000, 700000]
}
df = pd.DataFrame(data)

X = df[['SquareFootage', 'Bedrooms', 'Age']]
y = df['Price']

# Use a larger test split for demonstration or train on all data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print(f"Predicted Prices: {y_pred}")
print(f"Mean Squared Error: {mse}")

# Predict for new house
new_house = pd.DataFrame([[2500, 4, 12]], columns=['SquareFootage', 'Bedrooms', 'Age'])
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print(f"Predicted price for new house: ${predicted_price[0]:,.2f}")
