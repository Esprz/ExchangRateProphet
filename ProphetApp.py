import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.title("Exchange Rate Prophet")

currency = st.text_input("Enter the Exchange Rate code (e.g CADCNY=X)","CADCNY=X")
st.text("Some Common Exchange Rates Codes:")
st.text("CAD/CNY:CADCNY=X")
st.text("USD/CNY:CNY=X ")
st.text("USD/CAD:CAD=X")
st.text("...")

end = datetime.now()
start = datetime(end.year-5,end.month,end.day)
df = yf.download(currency, start,end)

model = load_model('exchange_rate_predict_model.keras')

st.subheader("Exchange Rate Data")
st.write(df)

splitting_len = int(len(df)*0.7)
x_test = pd.DataFrame(df.Close[splitting_len:])

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

period = 10
for i in range(period,len(scaled_data)):
    x_data.append(scaled_data[i-period:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = df.index[splitting_len+period:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([df.Close[:splitting_len+period],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)