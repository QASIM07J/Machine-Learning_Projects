import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load data
data = pd.read_csv("all_perth_310121.csv")

# Remove rows with NaN values
data = data.dropna()

# Define features and target
X = data[['BEDROOMS','BATHROOMS','GARAGE','FLOOR_AREA','LAND_AREA','LONGITUDE','LATITUDE','CBD_DIST']]
y = data['PRICE']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_prediction = model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_prediction)
print(f"Mean Squared Error: {mse}")

# Save the model to a file using pickle
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)