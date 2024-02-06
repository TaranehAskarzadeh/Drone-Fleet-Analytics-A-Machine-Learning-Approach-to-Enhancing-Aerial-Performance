def perform_multivariate_regression(X, y, title):
    # Generate polynomial features for multiple predictors
    poly_features_multi = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_multi = poly_features_multi.fit_transform(X)

    # Fit the model
    poly_model_multi = LinearRegression()
    poly_model_multi.fit(X_poly_multi, y)

    # Make predictions
    y_poly_pred_multi = poly_model_multi.predict(X_poly_multi)

    # Calculate metrics
    rmse_multi = np.sqrt(mean_squared_error(y, y_poly_pred_multi))
    r2_multi = r2_score(y, y_poly_pred_multi)

    print(f"{title} - RMSE: {rmse_multi}, R2: {r2_multi}")

# Assuming 'Rotors' is another predictor
X_multi = drone[['Payload', 'Rotors']]

# Multivariate Regression for Range
y_multi_range = drone['Range'].values
perform_multivariate_regression(X_multi, y_multi_range, "Multivariate Regression for Range")

# Multivariate Regression for Cruise
y_multi_cruise = drone['Cruise'].values
perform_multivariate_regression(X_multi, y_multi_cruise, "Multivariate Regression for Cruise")

# Multivariate Regression for Time
y_multi_time = drone['Time'].values
perform_multivariate_regression(X_multi, y_multi_time, "Multivariate Regression for Time")
