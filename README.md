Used Car Price Prediction

Project Overview

This project uses machine learning to estimate the asking prices of used cars based on their characteristics, including brand, model, manufacturing year, mileage, transmission, ownership history, and fuel type.

The project follows a data science workflow, from data cleaning and exploratory data analysis to feature engineering, model training, evaluation, and deployment through a Streamlit web application.

Objectives

Explore and clean used-car listing data.

Investigate factors that influence used-car asking prices.

Handle missing values and prepare data for machine learning.

Engineer features to improve predictive performance.

Train and evaluate a Random Forest regression model.

Deploy an interactive application for generating price estimates.

Technologies Used

Python — data processing and model development

Pandas and NumPy — data manipulation and numerical operations

Matplotlib and Seaborn — exploratory data analysis and visualization

Scikit-learn — preprocessing, model training, and evaluation

Joblib — saving and loading the trained model

Streamlit — interactive web application

Machine Learning Approach

A Random Forest Regressor was trained to predict the natural logarithm of car asking prices.

The workflow includes:

Data cleaning and preparation.

Exploratory data analysis to understand distributions and relationships.

Feature engineering, including kilometres driven per year.

One-hot encoding of categorical variables.

Standardization of selected numerical features.

Splitting the data into training and testing sets.

Training and evaluating the Random Forest model.

Converting predicted log-prices back to the original price scale using the exponential function.

Model Performance

The baseline Random Forest model achieved the following results on the test set:

Metric

Result

Mean Absolute Error (MAE)

0.2441

Root Mean Squared Error (RMSE)

0.3790

R-squared (R²)

0.8305

The R² score indicates that the model explains approximately 83% of the variation in the test target's log-transformed values. The MAE and RMSE are also measured on the log-price scale, not directly in the original currency.

Streamlit Application

The application allows users to enter a car's details and receive an estimated asking price.

The user can specify:

Brand and model

Manufacturing year

Kilometres driven

Transmission type

Previous ownership

Fuel type

The application loads the saved model and preprocessing tools from car_price_model.joblib to generate predictions.

Project Structure

used-car-price-prediction/
├── app.py
├── car_price_model.joblib
├── requirements.txt
├── README.md
└── notebooks/
    └── car_price_analysis.ipynb

The notebook filename and folder structure above are examples; adjust them to match the files in your repository.

Installation and Setup

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY

Replace the repository URL with your own GitHub repository URL.

2. Install dependencies

Ensure Python is installed, then run:

python -m pip install -r requirements.txt

3. Run the application

python -m streamlit run app.py

Streamlit will provide a local URL that you can open in your browser.

Limitations

Predictions depend on the quality and representativeness of the training data.

Asking prices may differ from final transaction prices.

Vehicle condition, location, service history, and other factors may affect market value but are not explicitly represented in the current model.

Estimates for unusual vehicles or combinations of features that are poorly represented in the training data may be less reliable.

The predictions should be treated as estimates rather than guaranteed market valuations.

Future Improvements

Improve missing-value handling and feature selection.

Compare additional regression algorithms and tune hyperparameters.

Investigate overfitting and improve generalization.

Add brand-dependent model selection to the application.

Improve input validation and support for unseen categories.

Deploy the application publicly using Streamlit Community Cloud.

Monitor model performance and retrain with updated market data.

Disclaimer

This project is intended for educational and portfolio purposes. The predictions are not professional vehicle valuations or financial advice.

Author

Lebohang Mahoa

Data Science | Marketing Analytics | Machine Learning

Live link: https://used-car-prediction-by-lebohangmahoa.streamlit.app/
