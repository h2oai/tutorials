import csv
import requests
import time
import datatable
from typing import List
from pathlib import Path

# Pump data to MLOps

# X is your input dt.Frame (https://github.com/h2oai/datatable)

# Download hydraulic_test_data.csv from Driverless AI Datasets Page
home = str(Path.home())
X = dt.Frame(home + "/Downloads/hydraulic_test_data.csv")

# This endpoint will need to be updated by copying your endpoint URL from current deployed model 
scoring_endpoint = "http://34.217.40.39:1080/85bfcf27-fbbb-4248-aa1d-49796c4bc328/model/score"

num_pumps = 1

X_pd = X.to_pandas()

for i in range(num_pumps):
    for row in range(X.nrows):
        print(list(X.names))
        print(list(list(X_pd.iloc[row].values)))
        response = requests.post(
            url = scoring_endpoint,
            json = {"fields": list(X.names), "rows": [[str(num) for num in list(X_pd.iloc[0].values)]]},
        )
        time.sleep(1)
        print(response)
        print(response.content)