# import pandas as pd
# import numpy as np
# from futu import OpenQuoteContext, KLType
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
#
#
# def fetch_stock_data(stock_symbol, start_date, end_date):
#     quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)
#     ret, data, _ = quote_context.request_history_kline(stock_symbol, start=start_date, end=end_date, ktype=KLType.K_DAY)
#     if ret == 0:
#         data = pd.DataFrame(data)
#     else:
#         print(f"Error: {data}")
#         data = None
#     quote_context.close()
#     return data
#
# def prepare_data(data):
#     data['returns'] = data['close'].pct_change()
#     data.dropna(inplace=True)
#     data['SMA_5'] = data['close'].rolling(window=5).mean()
#     data['SMA_20'] = data['close'].rolling(window=20).mean()
#     data.dropna(inplace=True)
#
#     X = data[['SMA_5', 'SMA_20']]
#     y = data['returns']
#
#     return X, y
#
#
# stock_symbol = 'HK.00700'
# start_date = '2021-01-01'
# end_date = '2021-12-31'
# data = fetch_stock_data(stock_symbol, start_date, end_date)
# X, y = prepare_data(data)
#
# model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# tscv = TimeSeriesSplit(n_splits=5)
# mse_scores = []
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mse_scores.append(mse)
#
# average_mse = np.mean(mse_scores)
# print(f'Average Mean Squared Error: {average_mse}')
#
# # Test on a different stock
# stock_symbol = 'HK.02318'
# data = fetch_stock_data(stock_symbol, start_date, end_date)
# X, y = prepare_data(data)
#
# tscv = TimeSeriesSplit(n_splits=5)
# mse_scores = []
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mse_scores.append(mse)
#
# average_mse = np.mean(mse_scores)
# print(f'Average Mean Squared Error for different stock: {average_mse}')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from futu import OpenQuoteContext, KLType

# Use the latest version of each library at the time of writing:
# pandas 1.3.3
# numpy 1.21.2
# torch 1.9.1
# scikit-learn 0.24.2
# futu-api 6.7.0

# Fetch stock data from FuTu API
from futu import OpenQuoteContext, KLType, KL_FIELD, RET_OK

def fetch_stock_data(stock_symbol, start_date, end_date):
    quote_context = OpenQuoteContext(host='127.0.0.1', port=11111)

    ret, data, _ = quote_context.request_history_kline(stock_symbol, start=start_date, end=end_date, ktype=KLType.K_DAY,
                                                    fields=[KL_FIELD.ALL])

    if ret == RET_OK:
        data = pd.DataFrame(data)
    else:
        print(f"Error: {data}")
        data = None

    quote_context.close()
    return data


# Preprocess data
def preprocess_data(data):
    data['returns'] = data['close'].pct_change()
    data.dropna(inplace=True)
    data['SMA_5'] = data['close'].rolling(window=5).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data.dropna(inplace=True)
    return data

# Split data into sequences for training and testing
# Split data into sequences for training and testing
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][0]
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # Convert to PyTorch tensors
    xs = torch.tensor(xs, dtype=torch.double)
    ys = torch.tensor(ys, dtype=torch.double)

    return xs, ys


# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parameters
stock_symbol = 'HK.00700'
start_date = '2020-01-01'
end_date = '2021-01-01'
seq_length = 20
test_size = 0.2
input_size = 4
hidden_size = 64
num_layers = 1
output_size = 1
learning_rate = 0.001
num_epochs = 100

# Fetch and preprocess data
data = fetch_stock_data(stock_symbol, start_date, end_date)
if data is not None:
    data = preprocess_data(data)

    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['returns', 'SMA_5', 'SMA_20', 'close']])
    data_scaled = pd.DataFrame(data_scaled, columns=['returns', 'SMA_5', 'SMA_20', 'close'], index=data.index)

    # Create sequences
    X, y = create_sequences(data_scaled.values, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = X_train.float()
    y_train = y_train.view(-1, 1).float()
    X_test = X_test.float()
    y_test = y_test.view(-1, 1).float()

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item()}')

# Fetch new data
new_stock_symbol = 'HK.09988'
new_start_date = '2022-01-01'
new_end_date = '2023-01-01'
new_data = fetch_stock_data(new_stock_symbol, new_start_date, new_end_date)
new_data = preprocess_data(new_data)

# Scale the new data
new_data_scaled = scaler.transform(new_data[['returns', 'SMA_5', 'SMA_20', 'close']])
new_data_scaled = pd.DataFrame(new_data_scaled, columns=['returns', 'SMA_5', 'SMA_20', 'close'], index=new_data.index)

# Create input sequences for the new data
new_sequences, _ = create_sequences(new_data_scaled.values, seq_length)

# Convert the input sequences to PyTorch tensors
new_sequences = torch.tensor(new_sequences).float()


# Make predictions using the trained model
model.eval()
with torch.no_grad():
    new_predictions = model(new_sequences)

# Convert the predictions to a NumPy array
new_predictions = new_predictions.numpy()

import matplotlib.pyplot as plt

# Convert the predictions back to the original scale
new_predictions_unscaled = scaler.inverse_transform(np.hstack((np.zeros((new_predictions.shape[0], 3)), new_predictions)))

# Calculate daily returns for actual and predicted prices
actual_returns = new_data['close'].pct_change().dropna().values[seq_length-1:]
predicted_returns = np.diff(new_predictions_unscaled[:, 3]) / new_predictions_unscaled[:-1, 3]

# Plot the actual daily returns and the predicted daily returns
plt.figure(figsize=(14, 6))
plt.plot(actual_returns, label='Actual Daily Returns')
plt.plot(predicted_returns, label='Predicted Daily Returns')
plt.xlabel('Days')
plt.ylabel('Daily Returns')
plt.title(f"Stock {new_stock_symbol} Daily Returns Prediction")
plt.legend()
plt.show()

# Get the latest seq_length days of data
latest_data = new_data[['returns', 'SMA_5', 'SMA_20', 'close']].values[-100:]

# Convert latest_data to a DataFrame with correct column names
latest_data = pd.DataFrame(latest_data, columns=['returns', 'SMA_5', 'SMA_20', 'close'])
#
# # Scale the input data
# latest_data_scaled = scaler.transform(latest_data)
#
# # Create input sequence and convert it to a PyTorch tensor
# input_sequence = torch.tensor(latest_data_scaled).unsqueeze(0).float()
#
# # Make a prediction using the trained model
# model.eval()
# with torch.no_grad():
#     tomorrow_prediction = model(input_sequence)
#
# # Convert the prediction to the original scale
# tomorrow_prediction_unscaled = scaler.inverse_transform(np.hstack((np.zeros((1, 3)), tomorrow_prediction)))
#
# # Calculate the predicted return
# predicted_return = (tomorrow_prediction_unscaled[0, 3] - new_data['close'].values[-1]) / new_data['close'].values[-1]
#
# print(f"Predicted return for stock {new_stock_symbol} tomorrow: {predicted_return * 100:.2f}%")

future_prices = []
for i in range(7):
    # Scale the input data
    latest_data_scaled = scaler.transform(latest_data)

    # Create input sequence and convert it to a PyTorch tensor
    input_sequence = torch.tensor(latest_data_scaled[-seq_length:, :], dtype=torch.float).unsqueeze(0)

    # Make a prediction using the trained model
    model.eval()
    with torch.no_grad():
        future_prediction = model(input_sequence)

    # Convert the prediction to the original scale
    future_prediction_unscaled = scaler.inverse_transform(np.hstack((np.zeros((1, 3)), future_prediction)))

    # Save the predicted future price
    future_prices.append(future_prediction_unscaled[0, 3])

    # Update the latest_data DataFrame with the new prediction
    new_row = {
        'returns': (future_prediction_unscaled[0, 3] - latest_data['close'].values[-1]) / latest_data['close'].values[
            -1],
        'SMA_5': latest_data['SMA_5'].values[-1],
        'SMA_20': latest_data['SMA_20'].values[-1],
        'close': future_prediction_unscaled[0, 3]}
    latest_data = pd.concat([latest_data, pd.DataFrame(new_row, index=[latest_data.index[-1] + 1])])

print("Predicted prices for the next week:", future_prices)

