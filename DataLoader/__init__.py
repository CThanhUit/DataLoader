import os
from os import path
from DataLoader.dataset_CICDDoS2019 import CICDDoS2019
from DataLoader.dataset_CICIDS2017 import CICIDS2017
from DataLoader.dataset_BoT_IoT import BoT_IoT
from DataLoader.dataset_ToN_IoT import ToN_IoT
from DataLoader.dataset_NetML import NetML
from DataLoader.dataset_N_BaIoT import N_BaIoT


# from pathlib import Path

# # project_dir = path.dirname(path.abspath(__file__))
# current_exec_dir = os.getcwd()

# if path.exists(os.path.join(current_exec_dir,'CICDDoS2019_clean_data_ver_30k_42seed.csv')) == False:
#     print("================================ Not found. Download demo data ================================")
    # import zipfile
    # with zipfile.ZipFile(os.path.join(project_dir,'DataLoader.zip'), 'r') as zip_ref:
    #     zip_ref.extractall(os.path.join(project_dir))
    # import urllib.request
    # urllib.request.urlretrieve("https://s3-hcm-r1.longvan.net/19414866-ids-datasets/CICDDoS2019/CICDDoS2019_clean_data_ver_30k_42seed.csv", "CICDDoS2019_clean_data_ver_30k_42seed.csv")
    # print("================================ End download data ================================")
