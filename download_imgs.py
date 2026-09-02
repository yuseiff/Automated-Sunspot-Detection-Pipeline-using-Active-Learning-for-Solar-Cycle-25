import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration - Adjust these values as needed
YEARS = ["2025"]
# MONTH = "07"
SAMPLING_INTERVAL_HOURS = 2  # Change this to adjust frequency (2 = every 2 hours)

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def download_images(url, save_path):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Collect all valid image links first
    images = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if 'Ic_flat_4k.jpg' in href:
            try:
                # Extract time from filename (format: YYYYMMDD_HHMMSS)
                time_str = href.split('_')[1]
                hour = int(time_str[:2])  # Get hour component
                images.append((hour, href))
            except (IndexError, ValueError):
                continue
    
    # Filter images to keep only those at regular intervals
    selected_images = []
    seen_hours = set()
    for hour, href in sorted(images, key=lambda x: x[0]):
        if hour % SAMPLING_INTERVAL_HOURS == 0 and hour not in seen_hours:
            selected_images.append(href)
            seen_hours.add(hour)
    
    # Download selected images
    for href in tqdm(selected_images, desc="Downloading images"):
        img_url = url + href
        img_data = requests.get(img_url).content
        with open(os.path.join(save_path, href), 'wb') as handler:
            handler.write(img_data)

def get_days_in_month(month_url):
    response = requests.get(month_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return sorted([link.get('href').strip('/') for link in soup.find_all('a') 
                if link.get('href').endswith('/') and link.get('href') not in ['../', '']])

def main():
    base_url = "https://jsoc1.stanford.edu/data/hmi/images/"
    
    # Validate year exists
    year_url = f"{base_url}{YEAR}/"
    if requests.get(year_url).status_code != 200:
        print(f"Invalid year: {YEAR}")
        return
    
    # Validate month exists
    month_url = f"{year_url}{MONTH}/"
    if requests.get(month_url).status_code != 200:
        print(f"Invalid month: {MONTH}")
        return
    
    # Get all days in the month
    days = get_days_in_month(month_url)
    print(f"Found {len(days)} days in {YEAR}-{MONTH}")
    # print(days[19:])
    for day in days:
        day_url = f"{month_url}{day}/"
        if requests.get(day_url).status_code == 200:
            save_path = os.path.join('Videos/Frames', YEAR, MONTH, day)
            create_directory(save_path)
            print(f"\nProcessing {YEAR}-{MONTH}-{day}")
            download_images(day_url, save_path)

if __name__ == "__main__":
    for YEAR in YEARS:
        print(f"Processing year: {YEAR}")
        for month in range(7,8):
            MONTH = f"{month:02d}"
            print(f"Processing month: {MONTH}")
            main()
    # main()