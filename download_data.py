import os
import json

# 1. Create the .kaggle directory if it doesn't exist
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

# 2. Extract your API key details
# Token: KGAT_89e42b205cd9a67300c9a9e0809c93a7
# Username for your Kaggle account (Set to your actual Kaggle username if known, or token format)
os.environ["KAGGLE_KEY"] = "KGAT_89e42b205cd9a67300c9a9e0809c93a7"

# Write token directly to local config so kaggle library finds it
token_file = os.path.join(kaggle_dir, 'access_token')
with open(token_file, 'w') as f:
    f.write("KGAT_89e42b205cd9a67300c9a9e0809c93a7")

# Alternative: Write legacy placeholder json to satisfy file requirement
json_file = os.path.join(kaggle_dir, 'kaggle.json')
if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
        json.dump({"username": "natalie", "key": "KGAT_89e42b205cd9a67300c9a9e0809c93a7"}, f)

from kaggle.api.kaggle_api_extended import KaggleApi

# 3. Authenticate with Kaggle
api = KaggleApi()
api.authenticate()

print("Kaggle API successfully authenticated!")

# 4. Download dataset into a local 'data' folder
api.dataset_download_files("innominate817/hagrid-sample-120k-384p", path="./data", unzip=True)

print("Dataset downloaded and extracted into ./data folder!")