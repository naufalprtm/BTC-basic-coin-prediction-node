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

## Components

- **Worker**: The node that publishes inferences to the Allora chain.
- **Inference**: A container that conducts inferences, maintains the model state, and responds to internal inference requests via a Flask application. This node operates with a basic linear regression model for price predictions.

## Setup
### Ensure you have Docker and Docker Compose installed. If not, you can install them using the instructions on the [Docker website](https://docs.docker.com/get-docker/) and [Docker Compose website](https://docs.docker.com/compose/install/).

  ```
sudo apt-get update -y && sudo apt-get upgrade -y

  ```


## Check GPU 
  ```
nvidia-smi
  ```

  ```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.76.01              Driver Version: 552.44         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     On  |   00000000:01:00.0  On |                  N/A |
|  0%   49C    P8             13W /  160W |    1435MiB /   8188MiB |      7%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A        24      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+

  ```

## Enable NVIDIA Runtime Globally: Ensure Docker uses the NVIDIA runtime by default. You can configure this by editing the Docker daemon configuration file /etc/docker/daemon.json:

  ```
sudo nano /etc/docker/daemon.json

  ```
  ```
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}


  ```
 ## plugin Docker For NVIDIA 
  ```
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

  ```


## Add GPU Support in docker-compose.yml: Modify the docker-compose.yml file to include GPU access for your containers:
  ```
services:
  inference:
    container_name: inference-basic-btc-pred
    build: .
    command: python -u /app/app.py
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/inference/BTC"]
      interval: 10s
      timeout: 5s
      retries: 12
    volumes:
      - ./inference-data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]


  ```
###  Model Initialization Errors: 
Ensure the correct device configuration (cuda or cpu) and required libraries are installed.
###  API Errors: 
Verify your API key for the CoinGecko API and ensure the URL is correct.
###  Data Processing Issues: 
Check the format of your downloaded files and ensure they are valid ZIP archives containing the expected data.

# Use 'auto' for device_map to handle device placement automatically
Update the in app.py:
  ```
device_map = "cuda"  # Change this according to your setup, such as 'cpu' or 'cuda'
  ```
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



### Steps to Setup

1. **Clone the Repository**
   
    ```
    
   git clone https://github.com/naufalprtm/BTC-basic-coin-prediction-node/tree/main && cd BTC-basic-coin-prediction-node

    ```
    
2. **Configuration**
    
    configuration file and populate it with your variables:
    ```sh
    config.json
    ```

      ```  
         {
                "wallet": {
                "addressKeyName": "allorachain",
                "addressRestoreMnemonic": "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
                "alloraHomeDir": "",
                "gas": "1000000",
                "gasAdjustment": 1.0,
                "nodeRpc": "https://sentries-rpc.testnet-1.testnet.allora.network/",
                "maxRetries": 1,
                "delay": 1,
                "submitTx": false
            },
                "worker": [
            {
                "topicId": 3,
                "inferenceEntrypointName": "api-worker-reputer",
                "loopSeconds": 5,
                "parameters": {
                "InferenceEndpoint": "http://localhost:8000/inference/{Token}",
                "Token": "BTC"
                }
            },
            {
                "topicId": 4, 
                "inferenceEntrypointName": "api-worker-reputer",
                "loopSeconds": 5,
                "parameters": {
                "InferenceEndpoint": "http://localhost:8000/inference/{Token}", 
                "Token": "BTC" 
               }
             }
           ]
        }
    ```
      
# Dockerfile

     
            # Stage 2: Build CUDA environment
            FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS cuda_env
      
      
#### *change this cuda:12.4.0-base-ubuntu22.04*
#### *ubuntu22.04 / ubuntu 20.04*
#### *cuda:12.4.0 find this from* ***NVDIA-smi***
    
       
            +-----------------------------------------------------------------------------------------+
            | NVIDIA-SMI 550.76.01              Driver Version: 552.44         CUDA Version: 12.4     |
            |-----------------------------------------+------------------------+----------------------+
    
       
- [Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags?page=6&page_size=&name=&ordering=)       


   ![Screenshot of the logs](https://github.com/naufalprtm/BTC-basic-coin-prediction-node/blob/main/Screenshot%202024-08-18%20234443.png)



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
   docker compose build && docker compose up -d && docker compose logs -f
    ```
    To confirm that the worker successfully sends the inferences to the chain, look for the following log:
    ```
    {"level":"debug","msg":"Send Worker Data to chain","txHash":<tx-hash>,"time":<timestamp>,"message":"Success"}
    ```
    
   ![Screenshot of the logs](https://github.com/naufalprtm/BTC-basic-coin-prediction-node/blob/main/Screenshot%202024-08-14%20011936.png)


   ![Screenshot of the logs](https://github.com/naufalprtm/BTC-basic-coin-prediction-node/blob/main/Screenshot%202024-08-15%20211315.png)


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


