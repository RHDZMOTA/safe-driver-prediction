from conf import settings
import requests
import os


def get_temporal_url(file_name, file_path):
    r = requests.get(settings.DROPBOX_DOWNLOAD.format(file_name=file_name, file_path=file_path))
    return r.json().get("url")


def download_file(cloud_file_name, cloud_file_path, local_filename):
    if os.path.exists(local_filename):
        return None
    temporal_url = get_temporal_url(cloud_file_name, cloud_file_path)
    r = requests.get(temporal_url)
    with open(local_filename, "wb") as file:
        file.write(r.content)
