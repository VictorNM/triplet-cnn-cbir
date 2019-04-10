import sys

sys.path.append('src')

from src.data.download_data import download_from_google_images

arguments = {
  "output_directory": "data/raw/google_images",
  "keywords": "wallet",
  "limit": 4,
  "size": "medium"
}

download_path = download_from_google_images(arguments=arguments)
print(download_path)
