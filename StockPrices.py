import pandas as pd
import talib
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split, GridSearchCV  # Added GridSearchCV import
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go  # Import Plotly

# Replace 'YOUR_ALPHA_VANTAGE_API_KEY' with your actual Alpha Vantage API key
api_key = '9O4VZ6CGP8T5SI17'
symbol = 'ADBE'

# Initialize Alpha Vantage API object
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch historical stock data from Alpha Vantage
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

# Sort the DataFrame by date
data.sort_index(inplace=True)

# Calculate technical indicators and features
data['SMA_10'] = data['4. close'].rolling(window=10).mean()
data['SMA_50'] = data['4. close'].rolling(window=50).mean()
data['SMA_200'] = data['4. close'].rolling(window=200).mean()
data['RSI'] = talib.RSI(data['4. close'], timeperiod=14)
macd, signal, _ = talib.MACD(data['4. close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = macd
upper, middle, lower = talib.BBANDS(data['4. close'], timeperiod=20)
data['BB_upper'] = upper
data['BB_middle'] = middle
data['BB_lower'] = lower
data['Volume_SMA_10'] = data['5. volume'].rolling(window=10).mean()
data['Volume_Ratio'] = data['5. volume'] / data['Volume_SMA_10']
data['Close_Lag_1'] = data['4. close'].shift(1)
data['Close_Lag_2'] = data['4. close'].shift(2)
data['Target'] = data['4. close'].shift(-1)

# Drop rows where the target variable is NaN
data.dropna(subset=['Target'], inplace=True)

# Select features and target variable
features = data[['SMA_10', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume_Ratio', 'Close_Lag_1', 'Close_Lag_2']]
target = data['Target']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define the hyperparameters grid to search
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a decision tree regressor
model = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Perform the grid search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = DecisionTreeRegressor(random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Make predictions with the best model
predictions = best_model.predict(X_test)

# Visualize the results using Plotly
fig = go.Figure()

# Actual stock prices
fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual Prices', line=dict(color='blue')))

# Predicted stock prices
fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted Prices', line=dict(color='red')))

# Customize the layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Stock Price',
    title='Actual vs. Predicted Stock Prices'
)

# Show the interactive plot
fig.show()

# Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) with the best model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Mean Absolute Error (MAE) with Best Model: {mae}")
print(f"Root Mean Squared Error (RMSE) with Best Model: {rmse}")

# Save the predictions to a DataFrame (optional)
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
predictions_df.to_excel('predictions_best_model.xlsx', index=True)

# Save the processed data to an Excel file (optional)
data.to_excel('output_data.xlsx', index=True)

# Save features to an Excel file for inspection (optional)
features.to_excel('features.xlsx', index=True)
