# House Price Prediction ML Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
# Assume CSV has columns like: 'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Price'
data = pd.read_csv('house_data.csv')

# Step 2: Inspect Data
print(data.head())
print(data.info())

# Step 3: Handle missing values
data = data.dropna()  # or use fillna for smarter imputation

# Step 4: Split features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split Train-Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 7: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)

# Step 9: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Optional: Predict new house price
# new_house = [[2500, 4, 3, 10]]  # Area, Bedrooms, Bathrooms, Age
# new_house_scaled = scaler.transform(new_house)
# price_prediction = model.predict(new_house_scaled)
# print("Predicted House Price:", price_prediction)
