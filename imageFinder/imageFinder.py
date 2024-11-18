import os
import requests
import json
import time
import zipfile

# Configurations
PIXABAY_API_URL = "https://pixabay.com/api/"
PIXABAY_API_KEY = "46884935-13c13d673b1d72d78cefb2173"  # Replace with your Pixabay API key
OUTPUT_DIR = "animal_images"
ZIP_FILE_NAME = "animal_images.zip"
DELAY_BETWEEN_REQUESTS = 1  # Reduce delay to speed up image downloads

# Function to search and download images by a search term using Pixabay API
def download_images(search_term):
    params = {
        'q': requests.utils.quote(search_term),
        'tags': 'animals,nature' if search_term not in ['girl', 'boy', 'person', 'child', 'woman', 'man', 'vampire'] else '',
        'key': PIXABAY_API_KEY,
        'image_type': 'photo',
        'per_page': 3,
            }
    download_and_save_images(params, search_term)

# Function to download and save images
def download_and_save_images(params, search_term):
    try:
        response = requests.get(PIXABAY_API_URL, params=params)
        response.raise_for_status()
        print(f"Full query URL: {response.url}")
        data = response.json()

        if 'hits' not in data or len(data['hits']) == 0:
            print(f"No images found for {search_term}")
            return

        for item in data['hits']:
            img_url = item['webformatURL']

            # Download and save image
            img_response = requests.get(img_url)
            img_response.raise_for_status()

            # Check if response content is an image
            if 'image' not in img_response.headers.get('Content-Type', ''):
                print(f"Invalid image content for {search_term}")
                continue

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, f"{search_term}.jpg")
            with open(output_path, 'wb') as f:
                f.write(img_response.content)

            print(f"Downloaded image for {search_term}")
            return
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {search_term}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {search_term}: {e}")

# Create ZIP file with all downloaded images
def create_zip():
    with zipfile.ZipFile(ZIP_FILE_NAME, 'w') as zipf:
        for folder_name, subfolders, filenames in os.walk(OUTPUT_DIR):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zipf.write(file_path, os.path.relpath(file_path, OUTPUT_DIR))

# Main script
def main():
    # Load animal names from JSON dataset
    with open('zoo_dataset.json', 'r') as f:
        animal_data = json.load(f)

    # Get a list of unique animal names
    animal_names = list(set([entry['animal_name'].lower() for entry in animal_data]))

    # Download images for each animal
    for animal in animal_names:
        print(f"Downloading image for {animal}...")
        download_images(animal)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Create a ZIP file of all downloaded images
    create_zip()
    print(f"All images downloaded and zipped into {ZIP_FILE_NAME}")

if __name__ == "__main__":
    main()
