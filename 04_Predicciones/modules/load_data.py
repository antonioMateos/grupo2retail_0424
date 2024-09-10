# librerias
import os
import pandas as pd

# Configura el directorio de trabajo
CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')

def load_data(data_path):
    path = os.path.join(DATA_PATH, data_path)
    return pd.read_csv(path)