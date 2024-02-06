# Drone Data Analysis Project

## Introduction
Exploring a comprehensive dataset of 45 drones to uncover insights into payload capacity, performance metrics, and key influencing features.

## Methodology
Utilized machine learning models including SVM, RF, Boosting algorithms, and KNN for payload prediction, alongside polynomial regression for analyzing payload's nonlinear impacts.

### Dataset Overview
The dataset comprises diverse specifications of drones. A brief overview is presented in the table below:

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

Correlation Plot.
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/7cfa290c-1ece-49e5-8bca-eaa0af919cd1)
## Visualizations
Polynomial Regression result plots.
   ![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/c2df6ebe-1bf9-433b-843b-b45243582beb)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/5040e5d7-a631-4a4c-a43b-57c86b7e79d9)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/ed0ee545-3158-4105-9739-d4c00cd5921f)
Multivariate Regression result plots.
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/f4e2ca32-23d0-4c3b-afae-bfc3643480b5)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/53b3f9fe-a638-4017-9e9a-690081fc4a86)
![image](https://github.com/TaranehAskarzadeh/Drone-Fleet-Analytics-A-Machine-Learning-Approach-to-Enhancing-Aerial-Performance/assets/65934906/00aa028e-9cab-40fc-81d6-ff5114100aab)



## Summary
The study has offered illuminating findings on two research fronts. Firstly, the comparative analysis of four machine learning models - Support Vector Machines (SVM), Random Forest (RF), Gradient Boosting, and K-Nearest Neighbors (KNN) - revealed that Gradient Boosting performed the best in predicting drone payload, considering three evaluation metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R-squared). Despite KNN possessing the highest R-squared value, suggesting the best fit among the models, its RMSE and MAE were higher than those of Gradient Boosting, thus indicating larger average prediction errors. Conversely, SVM showcased the least desirable performance, with the highest RMSE and MAE and the lowest R-squared value.
Secondly, the research unveiled valuable insights into the impact of payload on a drone's range, cruise speed, and flight time. The results suggested that a polynomial model of degree 2 seems to best capture the relationship between these variables. This infers that the effect of payload on these performance parameters is not linear but exhibits a curve, providing a nuanced understanding of drone performance under varying payload conditions. 


## How to Use
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Dive into the `notebooks/` directory for a comprehensive analysis.

## License
This project is under the MIT License - refer to the LICENSE file.
