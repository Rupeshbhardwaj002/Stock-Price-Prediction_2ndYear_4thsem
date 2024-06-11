import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor Website")

stock = st.text_input("Enter the Stock ID", "RELIANCE.NS")

end = datetime.now()
start = datetime(end.year-5, end.month, end.day)

input_dataset = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.h5")
st.subheader("Stock Data")
st.write(input_dataset)

splitting_len = int(len(input_dataset) * 0.7)
x_test = pd.DataFrame(input_dataset.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=input_dataset.index[splitting_len + 100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([input_dataset.Close[:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Predict the stock price for the next day
inputs = scaled_data[-100:].tolist()
input_data = np.array(inputs).reshape(1, -1, 1)
next_day_prediction = model.predict(input_data)

# Inverse transform the prediction
next_day_price = scaler.inverse_transform(next_day_prediction)

st.subheader(f"Price for today ({input_dataset.index[-1].date()}):")
st.write(input_dataset.Close[-1])

# Predict and display the price for the next day
next_day = input_dataset.index[-1] + pd.DateOffset(days=1)
st.subheader(f"Predicted price for the next day ({next_day.date()}):")
st.write(next_day_price[0][0])

# Get the investment amount from the user
investment = st.number_input("Enter the amount you want to invest tomorrow", min_value=0.0)

# Calculate the potential profit or loss
current_price = input_dataset.Close[-1]
predicted_return = next_day_price[0][0] / current_price - 1
potential_profit_loss = investment * predicted_return

# Display the potential profit or loss
st.subheader("Potential profit or loss for tomorrow")
st.write(potential_profit_loss)

st.subheader("MAY BE THE PORTFOLIO AFTER INVESTMENT(IF EVERYTHING GOES ACCORDING TO CALCULATIONS)")
st.write(investment + potential_profit_loss)

# Predict the stock price for the next day
inputs = scaled_data[-100:].tolist()
input_data = np.array(inputs).reshape(1, -1, 1)
next_day_prediction = model.predict(input_data)

# Inverse transform the prediction
next_day_price = scaler.inverse_transform(next_day_prediction)

# Calculate the percentage change
percentage_change = ((next_day_price - input_dataset.Close[-1]) / input_dataset.Close[-1]) * 100

# Display the percentage change
st.subheader("Predicted percentage change for tomorrow")
st.write(percentage_change)

forecast_days = st.selectbox("Select the prediction period", [7, 30, 60])


# Predict the stock prices for the next month
inputs = scaled_data[-30:].tolist()
for i in range(forecast_days):
    input_data = np.array(inputs[-30:]).reshape(1, -1, 1)
    prediction = model.predict(input_data)
    inputs.append(prediction[0])

# Inverse transform the predictions
forecasted_prices = scaler.inverse_transform(inputs[-forecast_days:])

# Create a new DataFrame for the forecasted prices
forecasted_df = pd.DataFrame(forecasted_prices, columns=['Forecasted Close'])

# Display the forecasted prices
st.subheader("Forecasted values for the next {} days".format(forecast_days))
st.write(forecasted_df)
