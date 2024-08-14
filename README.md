# Time Series Forecasting with Chronos and Flask

This repository provides a Flask-based application that leverages a time series forecasting model from Hugging Face's Chronos library to predict Bitcoin (BTC) prices. The application integrates with the CoinGecko API to fetch historical price data and performs inference based on this data. Additionally, the repository contains scripts for model training and data updating.

## Repository Origin

This code is based on repositories from the Allora Network:
This repository includes an example Allora network worker node designed to offer price predictions for BTC. The primary objective is to demonstrate the use of a basic inference model running within a dedicated container, showcasing its integration with the Allora network infrastructure to contribute valuable inferences.

- [Allora HuggingFace Walkthrough](https://github.com/allora-network/allora-huggingface-walkthrough)
- [Basic Coin Prediction Node](https://github.com/allora-network/basic-coin-prediction-node)

## Overview

The project consists of the following components:

- **Worker**: A node that publishes inferences to the Allora chain.
- **Inference**: A container that performs inferences, maintains the model state, and responds to internal inference requests via a Flask application. It uses a basic linear regression model for price predictions.
- **Updater**: A cron-like container designed to update the inference node's data by daily fetching the latest market information from Binance, ensuring the model remains current.

## Project Structure

The repository includes the following files:

- **`app.py`**: The Flask application that handles requests and performs time series forecasting.
- **`model.py`**: Script to download, format data, and train the model.
- **`updater.py`**: Utility functions for downloading data from Binance.
- **`config.py`**: Configuration file containing paths for data and models.

## Components

- **Worker**: The node that publishes inferences to the Allora chain.
- **Inference**: A container that conducts inferences, maintains the model state, and responds to internal inference requests via a Flask application. This node operates with a basic linear regression model for price predictions.
- **Updater**: A cron-like container designed to update the inference node's data by daily fetching the latest market information from Binance, ensuring the model stays current with new market trends.


## Setup

Prerequisites
Python 3.8+
Flask: 

  ```
pip install flask
  ```
Requests: 

  ```
pip install requests
  ```

Pandas: 

  ```
pip install pandas
  ```

Torch: 

  ```
pip install torch
  ```

Chronos: 

  ```
Install from Hugging Face's library
  ```

Scikit-Learn: 

  ```
pip install scikit-learn
  ```

###  Model Initialization Errors: 
Ensure the correct device configuration (cuda or cpu) and required libraries are installed.
###  API Errors: 
Verify your API key for the CoinGecko API and ensure the URL is correct.
###  Data Processing Issues: 
Check the format of your downloaded files and ensure they are valid ZIP archives containing the expected data.


###  app.py
This file contains the Flask application that serves as the main entry point for the API. It handles incoming requests, fetches data from the CoinGecko API, processes it using the ChronosPipeline for forecasting, and returns the result.

## API Key

Update the API key in app.py:
  ```
headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your API key
}
  ```

Check the `docker-compose.yml` file for the detailed setup of each component.

## Docker-Compose Setup

A complete working example is provided in the `docker-compose.yml` file.

### Steps to Setup

1. **Clone the Repository**
2. **Configuration**
    
    configuration file and populate it with your variables:
    ```sh
    config.json
    ```

3. **Initialize Worker**
    
    Run the following commands from the project's root directory to initialize the worker:
    ```sh
    chmod +x init.config
    ./init.config
    ```
    These commands will:
    - Automatically create Allora keys for your worker.
    - Export the needed variables from the created account to be used by the worker node, bundle them with your provided `config.json`, and pass them to the node as environment variables.

4. **Faucet Your Worker Node**
    
    You can find the offchain worker node's address in `./worker-data/env_file` under `ALLORA_OFFCHAIN_ACCOUNT_ADDRESS`. [Add faucet funds](https://docs.allora.network/devs/get-started/setup-wallet#add-faucet-funds) to your worker's wallet before starting it.

5. **Start the Services**
    
    Run the following command to start the worker node, inference, and updater nodes:
    ```sh
    docker compose up --build
    ```
    To confirm that the worker successfully sends the inferences to the chain, look for the following log:
    ```
    {"level":"debug","msg":"Send Worker Data to chain","txHash":<tx-hash>,"time":<timestamp>,"message":"Success"}
    ```

## Testing Inference Only

This setup allows you to develop your model without the need to bring up the offchain worker or the updater. To test the inference model only:

1. Run the following command to start the inference node:
    ```sh
    docker compose up --build inference
    ```
    Wait for the initial data load.

2. Send requests to the inference model. For example, request BTC price inferences:
    
    ```sh
    curl http://127.0.0.1:8000/inference/BTC
    ```
    Expected response:
    ```json
    {"value":"2564.021586281073"}
    ```

3. Update the node's internal state (download pricing data, train, and update the model):
    
    ```sh
    curl http://127.0.0.1:8000/update
    ```
    Expected response:
    ```sh
    0
    ```
    
## Docker Compose

### Building and Running the Services

#### Build the Docker images
```bash
docker-compose build
```

#### Start the services
```bash
docker-compose up -d
```

#### View the logs
```bash
docker-compose logs -f
```

#### Stop the services
```bash
docker-compose down
```


