import os
import json
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_FOLDER = os.environ.get("DATA_FOLDER")
RAW_DATA = os.environ.get("RAW_DATA")
DATASETS = os.environ.get("DATASETS")
SUBMIT = os.environ.get("SUBMIT")
DROPBOX_DOWNLOAD = os.environ.get("DROPBOX_DOWNLOAD")
TRAIN_LABEL = os.environ.get("TRAIN_LABEL")
TEST_LABEL = os.environ.get("TEST_LABEL")
VALIDATE_LABEL = os.environ.get("VALIDATE_LABEL")

GENERIC_CSV_FILENAME = os.environ.get("GENERIC_CSV_FILENAME")
GENERIC_PICKLE_FILENAME = os.environ.get("GENERIC_PICKLE_FILENAME")
MODEL_SETUP = os.environ.get("MODEL_SETUP")
TRAIN_DATA = os.environ.get("TRAIN_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
SAMPLE_SUBMISSION = os.environ.get("SAMPLE_SUBMISSION")
SUBMISSION_FILE = os.environ.get("SUBMISSION_FILE")
DATASETS_PICKLE = os.environ.get("DATASETS_PICKLE")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


DATA_PATH = join(PROJECT_DIR, DATA_FOLDER)
RAW_DATA_PATH = join(DATA_PATH, RAW_DATA)
DATASETS_PATH = join(DATA_PATH, DATASETS)
SUBMIT_PATH = join(DATA_PATH, SUBMIT)

class DataFilesConf:

    class Paths:
        data = DATA_PATH
        raw_data = RAW_DATA_PATH
        datasets = DATASETS_PATH
        submit = SUBMIT_PATH

    class FileNames:
        generic_filename_csv = join(DATA_PATH, GENERIC_CSV_FILENAME)
        model_setup = join(PROJECT_DIR, MODEL_SETUP)
        train = join(RAW_DATA_PATH, TRAIN_DATA)
        test = join(RAW_DATA_PATH, TEST_DATA)
        sample_submission = join(RAW_DATA_PATH, SAMPLE_SUBMISSION)
        submission_file = join(SUBMIT_PATH, SUBMISSION_FILE)
        datasets_pickle = join(DATASETS_PATH, DATASETS_PICKLE)


class ModelConf:

    class labels:
        train = TRAIN_LABEL
        test = TEST_LABEL
        validate = VALIDATE_LABEL

    @staticmethod
    def get_parameters(model_type, model_key):
        with open(DataFilesConf.FileNames.model_setup) as file:
            model_setup = json.load(file).get(model_type)
        return model_setup.get(model_key)


