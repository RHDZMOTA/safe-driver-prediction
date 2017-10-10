from conf.settings import DataFilesConf
from util.download import download_file
import os


def create_dir(reference_path):
    if not os.path.exists(reference_path):
        os.mkdir(reference_path)


def download_raw_data():
    download_file(
        cloud_file_name="sample_submission.csv",
        cloud_file_path="/kaggle/porto",
        local_filename=DataFilesConf.FileNames.sample_submission
    )
    download_file(
        cloud_file_name="test.csv",
        cloud_file_path="/kaggle/porto",
        local_filename=DataFilesConf.FileNames.test
    )
    download_file(
        cloud_file_name="train.csv",
        cloud_file_path="/kaggle/porto",
        local_filename=DataFilesConf.FileNames.train
    )


def create_dirs():
    create_dir(DataFilesConf.Paths.data)
    create_dir(DataFilesConf.Paths.raw_data)
    create_dir(DataFilesConf.Paths.datasets)
    create_dir(DataFilesConf.Paths.submit)
    create_dir("logs")


if __name__ == "__main__":
    create_dirs()
    download_raw_data()
