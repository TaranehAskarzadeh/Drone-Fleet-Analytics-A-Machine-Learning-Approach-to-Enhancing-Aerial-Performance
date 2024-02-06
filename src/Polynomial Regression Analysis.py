import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
drone = pd.read_csv("path/to/your/drone_dataset.csv")

def perform_polynomial_regression(X, y, title, x_label, y_label):
    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Fit the model
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Make predictions
    y_poly_pred = poly_model.predict(X_poly)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)

    print(f"{title} - RMSE: {rmse}, R2: {r2}")

    # Plot
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_poly_pred, color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Payload vs Range
X_range = drone[['Payload']].values
y_range = drone['Range'].values
perform_polynomial_regression(X_range, y_range, "Payload vs Range", "Payload", "Range")

# Payload vs Cruise
X_cruise = drone[['Payload']].values
y_cruise = drone['Cruise'].values
perform_polynomial_regression(X_cruise, y_cruise, "Payload vs Cruise", "Payload", "Cruise")

# Payload vs Time
X_time = drone[['Payload']].values
y_time = drone['Time'].values
perform_polynomial_regression(X_time, y_time, "Payload vs Time", "Payload", "Time")
