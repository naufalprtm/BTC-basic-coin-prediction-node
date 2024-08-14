import os
import pickle
from zipfile import ZipFile
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path
import numpy as np

# Define paths
binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
training_price_data_path = os.path.join(data_base_path, "btc_price_data.csv")

# ANSI color codes
RESET = "\033[0m"
INFO = "\033[94m"  # Blue
SUCCESS = "\033[92m"  # Green
WARNING = "\033[93m"  # Yellow
ERROR = "\033[91m"  # Red

def download_data():
    cm_or_um = "um"
    symbols = ["BTCUSDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path

    # Log download start
    print(f"{INFO}[INFO] Starting to download monthly Binance data for {symbols} in intervals {intervals} for years {years}.{RESET}")
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"{SUCCESS}[SUCCESS] Downloaded monthly data to {download_path}.{RESET}")

    # Log current time for daily data download
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    print(f"{INFO}[INFO] Starting to download daily Binance data for {symbols} for year {current_year}, month {current_month:02d}.{RESET}")
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"{SUCCESS}[SUCCESS] Downloaded daily data to {download_path}.{RESET}")

def format_data():
    # List and sort all files in binance_data_path
    files = sorted([x for x in os.listdir(binance_data_path)])
    print(f"{INFO}[INFO] Found {len(files)} files in {binance_data_path}.{RESET}")

    # Exit if no files are found
    if len(files) == 0:
        print(f"{ERROR}[ERROR] No files found to process. Exiting.{RESET}")
        return

    price_df = pd.DataFrame()

    # Process each file
    for file in files:
        zip_file_path = os.path.join(binance_data_path, file)
        print(f"{INFO}[INFO] Processing file: {zip_file_path}{RESET}")

        if not zip_file_path.endswith(".zip"):
            print(f"{WARNING}[WARNING] Skipping non-zip file: {file}{RESET}")
            continue

        with ZipFile(zip_file_path) as myzip:
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]

            # Assign column names and log
            df.columns = [
                "start_time", "open", "high", "low", "close", "volume",
                "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"
            ]
            print(f"{INFO}[INFO] Assigned columns to dataframe: {df.columns.tolist()}{RESET}")

            # Convert end_time to a proper datetime index
            df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])
            print(f"{INFO}[INFO] Appended data from {file} to main dataframe. Current dataframe shape: {price_df.shape}{RESET}")

    # Save the final dataframe to CSV
    price_df.sort_index().to_csv(training_price_data_path)
    print(f"{SUCCESS}[SUCCESS] Formatted data saved to {training_price_data_path}.{RESET}")

def train_model():
    # Load and log price data
    print(f"{INFO}[INFO] Loading price data from {training_price_data_path}.{RESET}")
    price_data = pd.read_csv(training_price_data_path)

    df = pd.DataFrame()

    # Convert date to timestamp and calculate average price
    print(f"{INFO}[INFO] Converting 'date' column to timestamp and calculating average price.{RESET}")
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)
    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Log data shapes
    print(f"{INFO}[INFO] Data after conversion and aggregation: {df.head()}{RESET}")

    # Reshape the data for regression
    x = df["date"].values.reshape(-1, 1)
    y = df["price"].values.reshape(-1, 1)
    print(f"{INFO}[INFO] Reshaped data: x.shape={x.shape}, y.shape={y.shape}{RESET}")

    # Split data into training and test sets
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)
    print(f"{INFO}[INFO] Training data: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}{RESET}")

    # Train the model
    print(f"{INFO}[INFO] Training the linear regression model.{RESET}")
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(f"{SUCCESS}[SUCCESS] Model training complete.{RESET}")

    # Validate model outputs
    try:
        # Generate predictions
        predictions = model.predict(x_train)
        
        # Check for invalid values
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise ValueError("Model predictions contain NaN or Inf values.")

        print(f"{SUCCESS}[SUCCESS] Model predictions are valid.{RESET}")
    except Exception as e:
        print(f"{ERROR}[ERROR] {str(e)}{RESET}")
        return

    # Ensure the model directory exists and save the model
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"{SUCCESS}[SUCCESS] Trained model saved to {model_file_path}.{RESET}")

