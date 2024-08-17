from flask import Flask, Response, request
import requests
import json
import pandas as pd
import torch
from chronos import ChronosPipeline
import time
import traceback
import subprocess
import ctypes
import numpy as np


# ANSI color codes
RESET = "\033[0m"
INFO = "\033[94m"  # Blue
SUCCESS = "\033[92m"  # Green
DEBUG = "\033[90m"  # Gray
ERROR = "\033[91m"  # Red

# Create our Flask app
app = Flask(__name__)

# Load the CUDA library
# Initialize CUDA library
use_cuda = True
try:
    cuda_lib = ctypes.CDLL("./worker.so")
    print(f"{SUCCESS}[SUCCESS] CUDA library loaded successfully.{RESET}")
except Exception as e:
    print(f"{ERROR}[ERROR] Failed to load CUDA library: {str(e)}{RESET}")
    use_cuda = False

# Define the Hugging Face model we will use
model_name = "amazon/chronos-t5-tiny"

# Detect if CUDA is available and set the device and dtype accordingly
device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and use_cuda else torch.float32

# Set up argument and return types for the CUDA function
cuda_lib.runMatrixMul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

print(f"Using device: {device}")

# Use 'auto' for device_map to handle device placement automatically
device_map = "cuda" if device == "cuda" else "cpu"  # Change this according to your setup, such as 'cpu' or 'cuda'

print(f"Device map: {device_map}")

# Print nvidia-smi output if CUDA is available
if device == "cuda":
    try:
        # Run the nvidia-smi command and capture its output
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("NVIDIA-SMI Output:")
        print(nvidia_smi_output)
    except subprocess.CalledProcessError as e:
        print(f"{ERROR}[ERROR] Error running nvidia-smi: {e}{RESET}")
        
def run_cuda_matrix_mul(A, B, N):
    C = np.zeros((N, N), dtype=np.float32)
    A_flat = A.flatten()
    B_flat = B.flatten()
    try:
        cuda_lib.runMatrixMul(A_flat, B_flat, C.flatten(), N)
    except Exception as e:
        print(f"{ERROR}[ERROR] CUDA function call failed: {str(e)}{RESET}")
        raise
    return C

@app.route("/run-cuda", methods=["POST"])
def run_cuda():
    try:
        data = request.json
        N = data.get("N")
        A = np.array(data.get("A"), dtype=np.float32)
        B = np.array(data.get("B"), dtype=np.float32)
        
        if A.shape != (N, N) or B.shape != (N, N):
            return {"error": "Matrices A and B must be NxN"}, 400
        
        result = run_cuda_matrix_mul(A, B, N)
        return {"result": result.tolist()}
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for a given token."""
    request_time = time.time()
    print(f"{INFO}[INFO] Received request at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}{RESET}")
    print(f"{INFO}[INFO] Request Details - IP: {request.remote_addr}, Method: {request.method}, Path: {request.path}{RESET}")
    print(f"{INFO}[INFO] Request Headers: {request.headers}{RESET}")

    if not token or token != "BTC":
        error_msg = "Token is required" if not token else "Token not supported"
        print(f"{ERROR}[ERROR] {error_msg}{RESET}")
        response = Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
        response_time = time.time()
        print(f"{INFO}[INFO] Sending response at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_time))}, Response Time: {response_time - request_time:.2f} seconds{RESET}")
        return response

    try:
        print(f"{INFO}[INFO] Initializing the ChronosPipeline with device_map=auto...{RESET}")

        # Load the model with the specified device and dtype
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,  # Ensure this is correctly set up
            torch_dtype=dtype
        )
        pipeline.model = pipeline.model.to_empty()
        # Ensure the model is properly initialized before moving it
        model = pipeline.model
        if device == "cuda":
           model = model.to("cuda")
        else:  
           model = model.to("cpu")
        
        # Log each parameter's shape and the device it's loaded on
        for param in pipeline.model.parameters():
            print(f"Param: {param.shape}, Device: {param.device}")
        print(f"{INFO}[INFO] Pipeline initialized successfully on {device} with dtype {dtype}.{RESET}")
    except Exception as e:
        print(f"{ERROR}[ERROR] Pipeline initialization failed: {str(e)}{RESET}")
        traceback.print_exc()
        response = Response(json.dumps({"pipeline error": str(e)}), status=500, mimetype='application/json')
        response_time = time.time()
        print(f"{INFO}[INFO] Sending response at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_time))}, Response Time: {response_time - request_time:.2f} seconds{RESET}")
        return response

    # Get the data from Coingecko
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXX"  # Replace with your API key
    }

    print(f"{INFO}[INFO] Requesting historical price data from Coingecko API...{RESET}")
    print(f"{DEBUG}[DEBUG] API URL: {url}{RESET}")
    print(f"{DEBUG}[DEBUG] API Headers: {headers}{RESET}")
    start_time = time.time()
    try:
        response = requests.get(url, headers=headers)
        api_response_time = time.time() - start_time
        print(f"{INFO}[INFO] API response time: {api_response_time:.2f} seconds{RESET}")
    except requests.exceptions.RequestException as e:
        print(f"{ERROR}[ERROR] API request failed: {str(e)}{RESET}")
        traceback.print_exc()
        response = Response(json.dumps({"error": "API request failed"}), status=500, mimetype='application/json')
        response_time = time.time()
        print(f"{INFO}[INFO] Sending response at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_time))}, Response Time: {response_time - request_time:.2f} seconds{RESET}")
        return response

    if response.status_code == 200:
        print(f"{INFO}[INFO] Data retrieved successfully from Coingecko.{RESET}")
        data = response.json()
        print(f"{DEBUG}[DEBUG] Raw data: {json.dumps(data, indent=2)[:500]}... [truncated]{RESET}")  # Truncate long data
        
        # Extracting and displaying detailed data
        try:
            df = pd.DataFrame(data["prices"], columns=["date", "price"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df = df[:-1]  # Remove the last row if needed
            
            # Extracting additional data from the Coingecko API response
            coin_data = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin", headers=headers).json()
            market_data = {
                "token_name": coin_data["name"],
                "symbol": coin_data["symbol"],
                "current_price": coin_data["market_data"]["current_price"]["usd"],
                "market_cap": coin_data["market_data"]["market_cap"]["usd"],
                "total_volume": coin_data["market_data"]["total_volume"]["usd"],
                "circulating_supply": coin_data["market_data"]["circulating_supply"],
                "max_supply": coin_data["market_data"]["max_supply"],
                "price_change_24h": coin_data["market_data"]["price_change_percentage_24h"],
                "last_updated": coin_data["last_updated"]
            }

            print(f"{INFO}[INFO] Processed DataFrame (last 5 rows):\n{df.tail(5)}{RESET}")
            print(f"{INFO}[INFO] Market Data:\n{json.dumps(market_data, indent=2)}{RESET}")

        except KeyError as e:
            print(f"{ERROR}[ERROR] Missing expected data key: {str(e)}{RESET}")
            print(f"{DEBUG}[DEBUG] Response data keys: {data.keys()}{RESET}")
            response = Response(json.dumps({"error": "Data processing error"}), status=500, mimetype='application/json')
            return response
    else:
        print(f"{ERROR}[ERROR] Failed to retrieve data from Coingecko API. Status Code: {response.status_code}{RESET}")
        print(f"{DEBUG}[DEBUG] API response: {response.text}{RESET}")
        response = Response(json.dumps({"Failed to retrieve data from the API": response.text}),
                            status=response.status_code,
                            mimetype='application/json')
        response_time = time.time()
        print(f"{INFO}[INFO] Sending response at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_time))}, Response Time: {response_time - request_time:.2f} seconds{RESET}")
        return response

    # Define the context and the prediction length
    context = torch.tensor(df["price"].values, dtype=torch.float32)  # Ensure the data type is correct

    # Check for invalid values in the context
    if torch.isnan(context).any() or torch.isinf(context).any():
        raise ValueError("Context tensor contains NaN or Inf values.")
    prediction_length = 1
    print(f"{INFO}[INFO] Context tensor created with shape: {context.shape}{RESET}")
    print(f"{DEBUG}[DEBUG] Context tensor content: {context}{RESET}")

    try:
        # Ensure tensor dtype matches model dtype
        if dtype == torch.bfloat16:
            context = context.to(dtype=torch.bfloat16)

        print(f"{INFO}[INFO] Generating forecast for BTC price in USDT using the model...{RESET}")
        prediction_start_time = time.time()
        forecast = pipeline.predict(context, prediction_length)  # Shape [num_series, num_samples, prediction_length]
        prediction_time = time.time() - prediction_start_time
        forecast_value = forecast[0].mean().item()
        print(f"{DEBUG}[DEBUG] Context tensor before prediction: {context}")
        print(f"{DEBUG}[DEBUG] Tensor min: {context.min()}, max: {context.max()}")
        # Log specific details about the forecast
        print(f"{SUCCESS}[SUCCESS] Forecast generated successfully.{RESET}")
        print(f"{INFO}[INFO] Predicted BTC/USDT value: {forecast_value:.2f} USDT{RESET}")
        print(f"{INFO}[INFO] Time taken for prediction: {prediction_time:.2f} seconds{RESET}")
    
        response = Response(str(forecast_value), status=200)
    except Exception as e:
        print(f"{ERROR}[ERROR] Error occurred during BTC price prediction: {str(e)}{RESET}")
        traceback.print_exc()
        response = Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    
    response_time = time.time()
    print(f"{INFO}[INFO] Sending response at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_time))}, Response Time: {response_time - request_time:.2f} seconds{RESET}")
    return response

# Run our Flask app
if __name__ == '__main__':
    start_time = time.time()
    print(f"{INFO}[INFO] Starting Flask app...{RESET}")
    print(f"{INFO}[INFO] Application started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}{RESET}")
    app.run(host="0.0.0.0", port=8000, debug=True)
