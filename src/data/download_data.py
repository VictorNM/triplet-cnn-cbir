"""
python file for download image from google image
"""

from google_images_download import google_images_download


def download_from_google_images(config=None, arguments=None):
  response = google_images_download.googleimagesdownload()
  absolute_path = response.download(arguments)
  return absolute_path
