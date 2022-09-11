import os
import pandas as pd
from pandas_profiling import ProfileReport

inputs_folder = os.path.join(os.environ["REPO_ROOT"], "input_data")
train_raw = pd.read_csv(os.path.join(inputs_folder, "train.csv"))
test_raw = pd.read_csv(os.path.join(inputs_folder, "test.csv"))
profile = ProfileReport(train_raw, title="Training Data Report")
profile.to_file("Training_data_report.html")
