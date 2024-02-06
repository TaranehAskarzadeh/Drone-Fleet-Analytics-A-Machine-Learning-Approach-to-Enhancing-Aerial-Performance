# Drone Data Analysis Project

## Introduction
This study will explore a drone dataset, meticulously compiled over a span of two years. The dataset comprises information on 45 distinct drones, with a variety of variables that encompass aspects such as the manufacturing company, model, drone type, number of rotors, electric motor count, drone dimensions (length, wingspan, and height), aspect ratio, maximum takeoff weight, payload capacity, range, maximum hover altitude, flight time, and cruise speed. These variables collectively provide a comprehensive overview of each drone's specifications and characteristics.
Building upon this wealth of information, this study intends to answer three crucial research questions. Firstly, it aims to identify the machine learning algorithm that can best predict a drone's payload capacity based on its features. Secondly, investigates how a drone's payload capacity impacts its range, cruise speed, and flight time, thereby analyzing the influence of payload on overall drone performance. Lastly, determines the key features that significantly impact the payload capacity of a drone. Through the execution of this research, I hope to glean valuable insights that can inform future drone designs and usage strategies.

## Methodology
### Research question 1: Which machine learning algorithm predicts the payload of drones better based on their features?
The research methodology comprises an assortment of machine learning models, data processing techniques, and statistical analyses to address the posed research questions. The chosen machine learning models are Support Vector Machines (SVM), Random Forest (RF), Boosting algorithms, and K-Nearest Neighbors (KNN). These models were chosen for their diversity in handling different data types, strengths in managing high dimensional data, robustness to outliers, and proficiency in capturing complex relationships within the data.
To address the first question, the four models will be trained and compared to determine which predicts drone payload most accurately based on drone features. This comparison will aid in identifying the model that can best capture and generalize the complex patterns within the drone dataset.
### Research question 2: How does a drone's payload affect its range, cruise speed, and flight time?
The second question seeks to understand the relationship between a drone's payload and its performance parameters. A nonlinear regression approach, specifically polynomial regression, will be employed to examine this possible nonlinear relationship. The advantage of using polynomial regression is its capability to model curvilinear relationships, which linear regression might miss.
Data preparation is a critical step in this process. The dataset will be imported into the R programming language, and its structure will be examined using the str() function. Necessary data cleaning processes, such as handling missing values and outliers, will be conducted to ensure the dataset's quality before proceeding to analysis.
A correlation analysis will be performed using the cor() function in R to explore pairwise relationships between variables. This step will provide insights into the strength and direction of relationships between various drone specifications.
I will also perform a multivariate regression analysis to evaluate the combined effect of multiple variables on the drone's range, cruise speed, and flight time.
Finally, we will conduct model comparisons using criteria such as the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) to identify the best fitting model while also considering model complexity. By implementing these diverse methodologies, we aim to gain a comprehensive understanding of the relationships within the drone data set.


## Dataset Overview
The dataset I provided contains information on 45 different drones, which is a part of my research over 2 years. Here is a brief overview of the variables present in the dataset:

| Attribute              | Description                                   |
|------------------------|-----------------------------------------------|
| **Company & Model**    | Manufacturer and model name                   |
| **Type & Rotors**      | Drone type and number of rotors               |
| **Dimensions**         | Length, wingspan, and height                  |
| **Performance Metrics**| MTOW, payload, range, flight time, and cruise speed |


## Results
Gradient Boosting emerged as the top model for payload prediction, supported by polynomial regression insights on payload's effect on performance.
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/c086361b-ef5e-416a-9a59-39a19d91292b)

| Model               | RMSE     | MAE      | R-squared |
|---------------------|----------|----------|-----------|
| SVM                 | 276.6410 | 251.7664 | 0.7318123 |
| Random Forest       | 192.5717 | 157.3051 | 0.8386109 |
| Gradient Boosting   | 132.9627 | 101.8924 | 0.9092537 |
| k-Nearest Neighbors | 179.3411 | 153.6775 | 0.9469411 |

These results present a comprehensive comparison of the four different machine learning models used to predict the payload of drones based on their features. The performance of these models was evaluated based on three key metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R-squared). Lower values of RMSE and MAE indicate better model performance, as they represent smaller average prediction errors. On the other hand, a higher R-squared value indicates a better fit of the model, as it represents the proportion of variance in the dependent variable that can be predicted from the independent variables.
The Gradient Boosting algorithm exhibits the best performance among all four models, with the lowest RMSE (132.96) and MAE (101.89) scores, indicating that it has the smallest average prediction errors. Although its R-squared value (0.909) is not the highest, it is still relatively high, suggesting that it can explain a significant proportion of the variance in the payload capacity of drones.
The Random Forest model also performs well, with an RMSE of 192.57 and an MAE of 157.31. Its R-squared value is 0.838, which is lower than Gradient Boosting and k-Nearest Neighbors but still represents a good fit.
The k-Nearest Neighbors (KNN) model has the highest R-squared value (0.946), indicating the best fit among the four models. However, its RMSE (179.34) and MAE (153.68) are higher than those of the Gradient Boosting model, suggesting it may have larger average prediction errors.
Lastly, the Support Vector Machine (SVM) model demonstrates the poorest performance among the four models, with the highest RMSE (276.64) and MAE (251.77), and the lowest R-squared value (0.731).
In conclusion, considering all three evaluation metrics, the Gradient Boosting model seems to perform best in predicting drone payload, followed by the KNN, Random Forest, and SVM models.

### Correlation Plot.
Perform correlation analysis to explore the pairwise relationships between the variables. Use functions like cor() or create correlation plots using ggplot2 to visualize the correlations. This helps in understanding the strength and direction of relationships between the variables.
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/7cfa290c-1ece-49e5-8bca-eaa0af919cd1)

## Visualizations
### Polynomial Regression result plots.

![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/c2df6ebe-1bf9-433b-843b-b45243582beb)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/5040e5d7-a631-4a4c-a43b-57c86b7e79d9)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/ed0ee545-3158-4105-9739-d4c00cd5921f)

Payload vs. Range: The polynomial regression model suggests that payload has a statistically significant linear effect on range (p-value = 0.0309 < 0.05), but the quadratic effect is not significant (p-value = 0.6563 > 0.05). This means that as the payload increases, the range of the drone is likely to change as well, although the relationship may not be a simple linear one. However, the model only explains about 10.99% of the variation in the range (R-squared = 0.1099), indicating that other factors not included in the model might have significant effects on range.
Payload vs. Cruise: The polynomial regression model suggests that payload has a significant linear effect on cruise speed (p-value = 7.18e-08 << 0.05). The quadratic term is not statistically significant (p-value = 0.152 > 0.05), suggesting that the relationship between payload and cruise speed is not necessarily quadratic. The model explains approximately 51.5% of the variation in the cruise speed (R-squared = 0.515), which is a significant proportion.
Payload vs. Time: The polynomial regression model suggests that neither the linear nor the quadratic effects of payload on flight time are statistically significant (p-value for both terms > 0.05). This suggests that the payload does not have a significant effect on flight time within the range of payload values in your dataset. The model explains only about 5.19% of the variation in flight time (R-squared = 0.0519), so other variables not included in the model might be more influential in determining flight time.
In summary, it appears that the drone's payload significantly affects its range and cruise speed but does not have a significant effect on its flight time, based on the dataset and the models used. 

### Multivariate Regression result plots.

![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/f4e2ca32-23d0-4c3b-afae-bfc3643480b5)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/53b3f9fe-a638-4017-9e9a-690081fc4a86)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/00aa028e-9cab-40fc-81d6-ff5114100aab)

The results of the multivariate regression analysis indicate the relationships between the predictor variables (Payload and Rotors) and the three response variables (Range, Cruise, and Time). Here is a discussion of the results in the context of the research question:
#### Range:
The intercept is 100.21, indicating the estimated Range when the Payload and Rotors are both zero.
The coefficient for poly(Payload, 2)1 is 125.32, suggesting a positive relationship between Payload (quadratic term) and Range. However, it is not statistically significant at the conventional significance level (p = 0.0248).
The coefficient for poly(Payload, 2)2 is 29.39, indicating a potential curvature in the relationship between Payload and Range. However, it is not statistically significant (p = 0.5868).
The coefficient for Rotors is -0.911, suggesting a negative relationship between Rotors and Range, but it is also not statistically significant (p = 0.3643).
The overall model is not highly significant based on the F-test (p = 0.1286), and the Adjusted R-squared is 0.06399, indicating that the predictors explain only a small proportion of the variability in Range.
#### Cruise:
The intercept is 124.78, representing the estimated Cruise value when Payload and Rotors are both zero.
The coefficient for poly(Payload, 2)1 is 203.66, indicating a positive relationship between Payload (quadratic term) and Cruise. It is statistically significant at a high significance level (p < 2e-16).
The coefficient for poly(Payload, 2)2 is -44.99, suggesting a potential curvature in the relationship between Payload and Cruise. However, it is not statistically significant (p = 0.164).
The coefficient for Rotors is -0.078, which is not statistically significant (p = 0.895), indicating no significant relationship between Rotors and Cruise.
The overall model is highly significant based on the F-test (p = 1.367e-06), and the Adjusted R-squared is 0.4797, indicating that the predictors explain a moderate proportion of the variability in Cruise.
##### Time:
The intercept is 51.42, representing the estimated Time value when Payload and Rotors are both zero.
The coefficient for poly(Payload, 2)1 is -9.14, suggesting a negative relationship between Payload (quadratic term) and Time, but it is not statistically significant (p = 0.765).
The coefficient for poly(Payload, 2)2 is 47.28, indicating a potential curvature in the relationship between Payload and Time. However, it is not statistically significant (p = 0.126).
The coefficient for Rotors is -0.592, which is not statistically significant (p = 0.297), indicating no significant relationship between Rotors and Time.
The overall model is not significant based on the F-test (p = 0.344), and the Adjusted R-squared is 0.00951, suggesting that the predictors have limited explanatory power for Time.
Overall, the multivariate regression analysis suggests that the relationship between Payload and the three response variables (Range, Cruise, and Time) is complex and may involve some degree of curvature. However, the relationships are not consistently statistically significant across the three response variables. The inclusion of Rotors as a predictor does not significantly contribute to explaining the variability in the response variables. 

### Model Comparison: 
Compared the performance of different models (linear, polynomial, multivariate) using model selection criteria such as Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC). These criteria help in identifying the model that best fits the data while considering the complexity of the model.
### Calculate AICc for each model
n <- nrow(drone)  # Sample size
k <- c(2, 3, 4, 5)  # Number of parameters for each model
logLik_values <- c(logLik(fit_linear), logLik(fit_poly2), logLik(fit_poly3), logLik(fit_poly4))  # Log-likelihood values
aics <- -2 * logLik_values + 2 * k + 2 * k * (k + 1) / (n - k - 1)
### Create a data frame to store AICc and BIC values
model_comparison <- data.frame(Model = c("Linear", "Poly2", "Poly3", "Poly4"),
                               AICc = aics,
                               BIC = bics)
print(model_comparison)
### Results:
| Model      | Type   | AICc    | BIC.df | BIC.BIC |
|------------|--------|---------|--------|---------|
| fit_linear | Linear | 486.7984| 3      | 493.9327|
| fit_poly2  | Poly2  | 488.8834| 4      | 497.5246|
| fit_poly3  | Poly3  | 490.5737| 5      | 500.6070|
| fit_poly4  | Poly4  | 492.0742| 6      | 503.3757|

The comparison includes linear and polynomial regression models and a multivariate regression model that considers the effects of payload and the number of rotors on the performance metrics.
Based on the AIC and BIC values:
Linear Model (fit_linear): This model assumes a linear relationship between the payload and the performance metrics. It has the lowest AIC and BIC values among all the models considered. This suggests that a linear relationship might be a reasonable approximation for understanding the impact of payload on range, cruise speed, and flight time. However, it is important to note that the linear model may not capture potential nonlinear relationships.
1.	Polynomial Models (fit_poly2, fit_poly3, fit_poly4): These models consider polynomial relationships of increasing degrees between the payload and the performance metrics. The AIC and BIC values increase with higher polynomial degrees, indicating that the additional complexity may not improve the model fit significantly. This suggests that a polynomial relationship of degree 2 (fit_poly2) might be sufficient to capture the nonlinear aspects of the payload's impact.
2.	Multivariate Model (fit_multivariate): This model incorporates both payload and the number of rotors as predictors of the performance metrics. However, the AIC and BIC values are not available for this model due to the limitations of the software used. Nevertheless, the coefficients of the multivariate model can provide insights into the combined effects of payload and the number of rotors on the performance metrics.
Overall, the results suggest that a polynomial model of degree 2 can be considered for analyzing the relationship between payload and the drone's range, cruise speed, and flight time. 

## Summary
The study has offered illuminating findings on two research fronts. Firstly, the comparative analysis of four machine learning models - Support Vector Machines (SVM), Random Forest (RF), Gradient Boosting, and K-Nearest Neighbors (KNN) - revealed that Gradient Boosting performed the best in predicting drone payload, considering three evaluation metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R-squared). Despite KNN possessing the highest R-squared value, suggesting the best fit among the models, its RMSE and MAE were higher than those of Gradient Boosting, thus indicating larger average prediction errors. Conversely, SVM showcased the least desirable performance, with the highest RMSE and MAE and the lowest R-squared value.
Secondly, the research unveiled valuable insights into the impact of payload on a drone's range, cruise speed, and flight time. The results suggested that a polynomial model of degree 2 seems to best capture the relationship between these variables. This infers that the effect of payload on these performance parameters is not linear but exhibits a curve, providing a nuanced understanding of drone performance under varying payload conditions. 


### How to Use
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Dive into the `notebooks/` directory for a comprehensive analysis.

### License
This project is under the MIT License - refer to the LICENSE file.
