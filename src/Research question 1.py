
# 2.1 Research question 1: Which machine learning algorithm predicts the payload of drones better based on their features?

# Support Vector Machine (SVM) Model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=500)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Gradient Boosting Model
gbm_model = GradientBoostingRegressor(n_estimators=5000, max_depth=4, learning_rate=0.2, loss='ls')
gbm_model.fit(X_train, y_train)
gbm_predictions = gbm_model.predict(X_test)

# k-Nearest Neighbors (KNN) Model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Evaluate models
def evaluate_model(predictions, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

svm_rmse, svm_mae, svm_r2 = evaluate_model(svm_predictions, y_test)
rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_predictions, y_test)
gbm_rmse, gbm_mae, gbm_r2 = evaluate_model(gbm_predictions, y_test)
knn_rmse, knn_mae, knn_r2 = evaluate_model(knn_predictions, y_test)

# Display results
print("Support Vector Machine (SVM) Model:")
print(f"RMSE: {svm_rmse:.4f}, MAE: {svm_mae:.4f}, R-squared: {svm_r2:.4f}")

print("\nRandom Forest Model:")
print(f"RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R-squared: {rf_r2:.4f}")

print("\nGradient Boosting Model:")
print(f"RMSE: {gbm_rmse:.4f}, MAE: {gbm_mae:.4f}, R-squared: {gbm_r2:.4f}")

print("\nk-Nearest Neighbors (KNN) Model:")
print(f"RMSE: {knn_rmse:.4f}, MAE: {knn_mae:.4f}, R-squared: {knn_r2:.4f}")



