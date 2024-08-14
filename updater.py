import os
import requests
from concurrent.futures import ThreadPoolExecutor

# ANSI color codes
RESET = "\033[0m"
INFO = "\033[94m"  # Blue
SUCCESS = "\033[92m"  # Green
WARNING = "\033[93m"  # Yellow
ERROR = "\033[91m"  # Red

# Function to download the URL, called asynchronously by several child processes
def download_url(url, download_path):
    target_file_path = os.path.join(download_path, os.path.basename(url))
    
    if os.path.exists(target_file_path):
        print(f"{INFO}[INFO] File already exists, skipping download: {url}{RESET}")
        return
    
    print(f"{INFO}[INFO] Downloading URL: {url}{RESET}")
    try:
        response = requests.get(url)
        if response.status_code == 404:
            print(f"{WARNING}[WARNING] File not found (404): {url}{RESET}")
        else:
            # Create the entire path if it doesn't exist
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

            with open(target_file_path, "wb") as f:
                f.write(response.content)
            print(f"{SUCCESS}[INFO] Successfully downloaded: {url} to {target_file_path}{RESET}")
    except Exception as e:
        print(f"{ERROR}[ERROR] Failed to download {url}. Exception: {str(e)}{RESET}")

def download_binance_monthly_data(cm_or_um, symbols, intervals, years, months, download_path):
    print(f"{INFO}[INFO] Starting Binance monthly data download for {cm_or_um}.{RESET}")
    
    # Verify if cm_or_um is correct, if not, exit
    if cm_or_um not in ["cm", "um"]:
        print(f"{ERROR}[ERROR] CM_OR_UM can only be 'cm' or 'um'. Invalid value provided.{RESET}")
        return
    
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/monthly/klines"
    print(f"{INFO}[INFO] Base URL for downloads: {base_url}{RESET}")

    # Main loop to iterate over all the arrays and launch child processes
    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                for year in years:
                    for month in months:
                        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip"
                        print(f"{INFO}[INFO] Scheduling download for {url}{RESET}")
                        executor.submit(download_url, url, download_path)
    
    print(f"{INFO}[INFO] Completed scheduling Binance monthly data downloads.{RESET}")

def download_binance_daily_data(cm_or_um, symbols, intervals, year, month, download_path):
    print(f"{INFO}[INFO] Starting Binance daily data download for {cm_or_um}.{RESET}")
    
    if cm_or_um not in ["cm", "um"]:
        print(f"{ERROR}[ERROR] CM_OR_UM can only be 'cm' or 'um'. Invalid value provided.{RESET}")
        return
    
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/daily/klines"
    print(f"{INFO}[INFO] Base URL for downloads: {base_url}{RESET}")

    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                for day in range(1, 32):  # Assuming days range from 1 to 31
                    url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
                    print(f"{INFO}[INFO] Scheduling download for {url}{RESET}")
                    executor.submit(download_url, url, download_path)
    
    print(f"{INFO}[INFO] Completed scheduling Binance daily data downloads.{RESET}")
