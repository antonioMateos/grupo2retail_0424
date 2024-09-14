import os
import pandas as pd

def get_data_path():
    # Detecta si está en Google Colab
    if 'COLAB_GPU' in os.environ:
        base_path = '/content/drive/MyDrive/TFM_Retail_Repo/_data'  # Ruta en Google Colab
    else:
        base_path = os.path.join(os.path.dirname(os.getcwd()), '_data') # Ruta local
    
    # Verifica si la carpeta _data existe
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No se encontró la carpeta _data en {base_path}")
    
    return base_path

# Función para obtener la ruta completa del archivo concatenando
def get_file_path(file_name):
    return os.path.join(get_data_path(), file_name)

def get_data(file_name):
    file_path = get_file_path(file_name)
    return pd.read_csv(file_path)

# # Ejemplo de uso:
# file_path = get_file_path('archivo.csv')
# print(f"Ruta del archivo: {file_path}")

# Puedes cargar datos usando pandas, por ejemplo:
# import pandas as pd
# df = pd.read_csv(file_path)
