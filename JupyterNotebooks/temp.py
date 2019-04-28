import os
import tarfile
import pandas as pd
from six.moves import urllib
import matplotlib as plt
#import tensorflow as tf

download_root="https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH="datasets/housing"
housing_url=download_root+HOUSING_PATH+"/housing.tgz"

def fetch_housing_data(housing_url=housing_url,HOUSING_PATH=HOUSING_PATH):
    if not os.path.isdir(HOUSING_PATH):
        os.makedirs(HOUSING_PATH)
    tgz_path= os.path.join(HOUSING_PATH, "housing.tgz")
    #print(tgz_path)
    #urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()


def load_housing_data():
    csv_path=os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing=load_housing_data()
housing.head()
housing.
