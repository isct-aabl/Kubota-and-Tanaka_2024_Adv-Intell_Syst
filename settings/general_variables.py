from dotenv import load_dotenv
import os

import yaml

"""
setting_yaml_file_base だけファイルに合わせて変更
"""
setting_yaml_file_base = "Wing_A"
# setting_yaml_file_base = "Wing_B"
# setting_yaml_file_base = "Wing_A-2"

with open(f"/home/user/src/scripts/settings/config/{setting_yaml_file_base}.yaml", 'r', encoding='utf-8') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

DATA_VERSIONS = config["DATA_VERSIONS"]
DATA_VERSION_FOLDER_NAME = config["DATA_VERSION_FOLDER_NAME"]
SG_COLORS = config["SG_COLORS"]
EN_COLORS = config["EN_COLORS"]
CLASSES_COLORS = config["CLASSES_COLORS"]
DATA_COLUMN_NAMES = config["DATA_COLUMN_NAMES"]
WIND_ENVIRONMENTS = config["WIND_ENVIRONMENTS"]
ENCODER_RESOLUTION = int(config["ENCODER_RESOLUTION"])
FLAPPING_FREQUENCY = int(config["FLAPPING_FREQUENCY"])
FREQUENCY_S = int(config["FREQUENCY_S"])
TIME_S = int(config["TIME_S"])
DATA_LOGGER_TYPE = config["DATA_LOGGER_TYPE"]
NO_FLAPPING_DATA_FILE_NAME = config["NO_FLAPPING_DATA_FILE_NAME"]
MODEL_FILE_NAME = config["MODEL_FILE_NAME"]

# WARNING
DEPRECIATING_WARNING = "#####################\n\nThis mode is depreciated\n\n#####################\n"
IMPLEMENTATION_ORDER_WARNING = "After checking the graph of raw data (arranged_strain_dataframes), " \
                               "implement get_split_strain_data_dfs."
IMPLEMENTATION_LOAD_DATA_WARNING = "Please implement cnn_data_handler.load_train_test_data(should_normalize=True/False)."
NO_FILE_CHOSEN_WARNING = "No file's been chosen."

# QUESTION PROMPT
QUESTION_PROMPT_WHOLE_OR_SPLIT = "Would you like whole data or split data?\n1: Whole data\n2: Split data"
QUESTION_PROMPT_FLATTEN_OR_RAW = "Would you like flatten data or raw data?\n1: Flatten data\n2: Raw data"
QUESTION_PROMPT_NORMALIZE_DATA = "Would you like normalized data?\ny: yes\nn: no"
QUESTION_PROMPT_SPLIT_NUM = "Which split data would you like? Pick one from below please."
QUESTION_PROMPT_FLATTENING_PARAMETER = "Which flattening parameter data would you like? Pick One from below please."
QUESTION_PROMPT_WIND_INFO = "Which wind direction and speed data are you interested in? " \
                            "Please Pick from the options below."
QUESTION_PROMPT_TARGET_DATA_INDEX = "Which position of the data are you interested in? Please pick from the list below."
QUESTION_PROMPT_NORMALIZE_ML_DATA = "Do you want to standardize the training data and the test data? \ny: yes, n: no"
QUESTION_PROMPT_END = 'If you would like to end, please type "end".'
QUESTION_PROMPT_WHICH_CHANNEL = "Which channel would you like to use? Pick one from the list below."
QUESTION_PROMPT_WHICH_COLUMN = "Which channel of data would you like? \n" \
                               f"Please select from 1-{len(DATA_COLUMN_NAMES)}, or any combination of them. " \
                               "If you want all, simply hit Enter."

# CHECK RESULT PROMPT

# FOLDER NAME
FLAPPING_DATA_FOLDER_NAME = "flapping"
RAW_DATA_FOLDER_NAME = "raw_data"
FLATTEN_DATA_FOLDER_NAME = "flatten_data"
NORMALIZED_DATA_FOLDER_NAME = "normalized"
NOT_NORMALIZED_DATA_FOLDER_NAME = "not_normalized"

# FILE/FOLDER PREFIX
CHANNEL_FILE_PREFIX = "CH"
MODEL_FILE_PREFIX = "my_model"
BEST_MODEL_FILE_PREFIX = "my_best_model"
MODEL_ACCURACY_FILE_PREFIX = "accuracy"
MODEL_HISTORY_FILE_PREFIX = "history"

# FILE/FOLDER SUFFIX
MODEL_RESULT_FOLDER_SUFFIX = "result"
MODEL_ACCURACY_FILE_NUM_SUFFIX = "num"
MODEL_ACCURACY_FILE_PERCENTAGE_SUFFIX = "percentage"

# Paths
ORIGINAL_DATA_PATHS = [rf"../data/{data_version}/1_original_data" for data_version in DATA_VERSIONS]
SPLIT_DATA_PATH = rf"../data/{DATA_VERSION_FOLDER_NAME}/2_split_data"
TRAINING_FILE_PATH = rf"../data/{DATA_VERSION_FOLDER_NAME}/3_training_data"
MODEL_FILE_PATH = rf"../models/{DATA_VERSION_FOLDER_NAME}/{MODEL_FILE_NAME}"
TENSORFLOW_LOG_FILE_PATH = rf"../logs/{DATA_VERSION_FOLDER_NAME}"
