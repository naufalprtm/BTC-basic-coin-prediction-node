import requests
import sys
import json

# ANSI color codes
RESET = "\033[0m"
INFO = "\033[94m"  # Blue
ERROR = "\033[91m"  # Red

def process(argument):
    """Process the inference request with the provided argument."""
    headers = {'Content-Type': 'application/json'}
    url = f"http://inference:8000/inference/{argument}"
    print(f"{INFO}[INFO] Sending request to URL: {url}{RESET}")
    print(f"{INFO}[INFO] Request headers: {headers}{RESET}")
    
    try:
        response = requests.get(url, headers=headers)
        print(f"{INFO}[INFO] Received response with status code: {response.status_code}{RESET}")
        
        if response.status_code == 200:
            print(f"{INFO}[INFO] Response content: {response.text}{RESET}")
        else:
            print(f"{ERROR}[ERROR] Failed to retrieve valid response. Status Code: {response.status_code}, Response: {response.text}{RESET}")
        
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"{ERROR}[ERROR] Request exception occurred: {str(e)}{RESET}")
        raise
    except Exception as e:
        print(f"{ERROR}[ERROR] General exception occurred during request: {str(e)}{RESET}")
        raise

if __name__ == "__main__":
    print(f"{INFO}[INFO] Script started with arguments: {sys.argv}{RESET}")
    
    try:
        if len(sys.argv) < 5:
            error_msg = f"{ERROR}[ERROR] Not enough arguments provided: {len(sys.argv)-1}, expected 4 arguments: topic_id, blockHeight, blockHeightEval, default_arg{RESET}"
            print(error_msg)
            value = json.dumps({"error": error_msg})
        else:
            topic_id = sys.argv[1]
            blockHeight = sys.argv[2]
            blockHeightEval = sys.argv[3]
            default_arg = sys.argv[4]
            
            print(f"{INFO}[INFO] Parsed arguments - topic_id: {topic_id}, blockHeight: {blockHeight}, blockHeightEval: {blockHeightEval}, default_arg: {default_arg}{RESET}")
            
            response_inference = process(argument=default_arg)
            response_dict = {"infererValue": response_inference}
            value = json.dumps(response_dict)
            print(f"{INFO}[INFO] Final output: {value}{RESET}")
    
    except Exception as e:
        error_msg = f"{ERROR}[ERROR] Unhandled exception occurred: {str(e)}{RESET}"
        print(error_msg)
        value = json.dumps({"error": error_msg})
    
    print(f"{INFO}[INFO] Script completed with output: {value}{RESET}")
