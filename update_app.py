import os
import requests

# Get the inference API address from the environment variables
inference_address = os.environ.get("INFERENCE_API_ADDRESS")
if not inference_address:
    print("[ERROR] Environment variable 'INFERENCE_API_ADDRESS' is not set.")
    exit(1)

url = f"{inference_address}/update"
print(f"[INFO] Sending request to update inference worker data at: {url}")

try:
    response = requests.get(url)
    if response.status_code == 200:
        print("[INFO] Request successful. Processing response...")
        content = response.text

        if content == "0":
            print("[INFO] Response content is '0'. Exiting with status code 0.")
            exit(0)
        else:
            print(f"[WARNING] Unexpected response content: '{content}'. Exiting with status code 1.")
            exit(1)
    else:
        print(f"[ERROR] Request failed with status code: {response.status_code}. Exiting with status code 1.")
        exit(1)
except requests.exceptions.RequestException as e:
    print(f"[ERROR] Exception occurred during the request: {str(e)}. Exiting with status code 1.")
    exit(1)
