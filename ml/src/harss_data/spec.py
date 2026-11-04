"""Frozen conventions shared by data readers and future model code."""

CONTRACT_VERSION = "preprocessing-v1"
CHANNELS = (
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
)
UNITS = ("g", "g", "g", "rad/s", "rad/s", "rad/s")
CLASSES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)
OFFICIAL_TRAIN_SUBJECTS = (1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21,
                           22, 23, 25, 26, 27, 28, 29, 30)
OFFICIAL_TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)
WINDOW_SIZE = 128
STRIDE = 64
SAMPLE_HZ = 50
PERIOD_NS = 20_000_000
MAX_GAP_NS = 100_000_000
STANDARD_GRAVITY = 9.80665
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
WRAPPER_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
DATASET_SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"
