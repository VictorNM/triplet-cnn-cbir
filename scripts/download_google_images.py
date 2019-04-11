import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.download_data import download_from_google_images


def download(keywords, arguments):
    for keyword in keywords:
        arguments['keywords'] = keyword
        download_from_google_images(arguments=arguments)


if __name__ == '__main__':
    # https://github.com/hardikvasa/google-images-download/blob/master/README.rst#arguments
    arguments = {
        "output_directory": "data/raw/small_objects",
        "limit": 100,
        "size": "medium",
        "format": "jpg",
        "type": "photo"
    }
    keywords = ['knife', 'scissor', 'watch', 'wallet', 'glasses']
    download(keywords, arguments)