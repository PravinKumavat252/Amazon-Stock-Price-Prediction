# Amazon-Stock-Price-Prediction

This project utilizes Machine Learning (ML) techniques to predict the stock price of Amazon (AMZN) using historical data. The model is built using Long Short-Term Memory (LSTM) networks, which are a type of Recurrent Neural Network (RNN) that is particularly effective for sequential data like stock prices.

## Project Overview
This repository contains a machine learning-based solution to predict Amazon's stock price using historical data. The model uses a variety of libraries, such as pandas, numpy, matplotlib, seaborn, tensorflow, and plotly to preprocess, train, and evaluate the model.

## Key Features
* Data Collection: Historical stock price data is fetched and preprocessed.
* Data Preprocessing: The stock price data is normalized using MinMax scaling to improve model performance.
* Model: A LSTM model is used for prediction as it is well-suited for time-series forecasting.
* Evaluation: The performance of the model is evaluated using Mean Squared Error (MSE).
* Visualization: Data visualizations of stock prices are presented using matplotlib, seaborn, and interactive charts with plotly.
## Requirements
Before running this project, make sure you have the necessary libraries installed. You can install them using pip:

## Files
* amazon_stock_prediction.py: The main script containing the code to load data, preprocess, train the model, and make predictions.
* requirements.txt: List of dependencies for the project.
* README.md: Documentation for the project.

## Detailed Overview of the Code
#### 1. Data Collection and Preprocessing:

* The dataset is loaded using pandas. Historical stock prices are sourced from Yahoo Finance or a similar API.
* Data is preprocessed by scaling the prices using MinMaxScaler to normalize the data.
* The dataset is split into training and testing sets.
  
#### 2. Building the Model:

* A Sequential LSTM model is built using tensorflow.keras.
* The model contains LSTM layers, Dropout layers to prevent overfitting, and a Dense layer for output prediction.
* The model is compiled using the Adam optimizer and Mean Squared Error as the loss function.
#### 3. Training the Model:

* The model is trained on the training data and evaluated on the test set.
#### 4. Prediction and Visualization:

* Predictions are made for the test dataset.
* Real vs predicted stock prices are visualized using matplotlib and plotly for interactive plots.
## Example Output
Upon successful execution, you will see graphs showing the predicted and actual stock prices over time. The output will include the Mean Squared Error (MSE), which gives an indication of the model’s performance.

## Conclusion
The LSTM model provides a useful approach to forecasting time-series data like stock prices. While predictions may not always be accurate due to the volatility of the stock market, this model offers a starting point for understanding and experimenting with machine learning for stock price prediction.

## Future Work
* Improving Accuracy: Further tuning of the model and testing with additional features (e.g., technical indicators, news sentiment, etc.).
* Model Optimization: Experimenting with more advanced models like GRU (Gated Recurrent Unit) or attention mechanisms.
* Real-Time Prediction: Implementing the model to make real-time predictions using live stock data.
