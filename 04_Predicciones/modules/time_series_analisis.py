# Librerias
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns

# TimeSeries
from statsmodels.tsa.seasonal import seasonal_decompose

# Config
DATE_COL = 'fecha'
SALES_COL = 'cifra'

def filter_data(data, col, value):
    return data[data[col] == value]

def plot_sales(data, col):
    """
    Visualiza la evolución de las ventas agrupadas por la columna especificada.
    
    Args:
    data (DataFrame): El DataFrame que contiene los datos de ventas.
    col (str): El nombre de la columna por la que se desea agrupar las ventas.
    
    """
    # Asegurarse de que la columna 'fecha' esté en formato datetime
    data['fecha'] = pd.to_datetime(data[DATE_COL])

    # Agrupar por la columna elegida y por la fecha, y sumar las ventas (cifra)
    sales_by_col = data.groupby([col, DATE_COL])[SALES_COL].sum().reset_index()

    plt.figure(figsize=(14, 8))

    # Graficar la evolución de las ventas para cada valor único en la col
    for valor in sales_by_col[col].unique():
        datos = sales_by_col[sales_by_col[col] == valor]
        plt.plot(datos[DATE_COL], datos[SALES_COL], label=valor, linewidth=1)

    # Configuración del gráfico
    plt.title(f"Evolución de ventas por {col}")
    plt.xlabel(DATE_COL)
    plt.ylabel(f'Ventas ({SALES_COL})')
    plt.legend(title=col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def decompose_time_series(data, date_col=DATE_COL, target_col=SALES_COL, freq='D', model='additive', period=None):
    """
    Descompone la serie temporal en tendencia, estacionalidad y ruido.
    
    Args:
    data (DataFrame): El DataFrame que contiene los datos.
    date_col (str): El nombre de la columna que contiene las fechas.
    target_col (str): El nombre de la columna de la serie temporal que quieres descomponer.
    freq (str, opcional): La frecuencia de los datos, por ejemplo, 'W' (semanal), 'M' (mensual), etc.
    model: tipo de modelo => additive or multiplicative
    period: periodo en el que a priori se produce estacionalidad
    
    Returns:
    None: Muestra la descomposición gráfica de la serie temporal.
    """
    # Asegurarse de que la columna de fecha esté en formato datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Establecer la columna de fecha como el índice
    data = data.set_index(date_col)

    # Asegurarse de que los datos estén ordenados por la columna de fecha
    data = data.sort_index()

    # Resamplear los datos a la frecuencia deseada si es necesario
    data_resample = data[target_col].resample(freq).sum()

    # Descomponer la serie temporal
    descomposicion = seasonal_decompose(data_resample, model=model, period=period)

    # Configurar el tamaño de la figura y crear subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Graficar la serie original
    axs[0].plot(data_resample.index, data_resample.values)
    axs[0].set_title('Serie Original')
    
    # Graficar la tendencia
    axs[1].plot(descomposicion.trend.index, descomposicion.trend.values)
    axs[1].set_title('Tendencia')
    
    # Graficar la estacionalidad
    axs[2].plot(descomposicion.seasonal.index, descomposicion.seasonal.values)
    axs[2].set_title('Estacionalidad')
    
    # Graficar el ruido
    axs[3].plot(descomposicion.resid.index, descomposicion.resid.values)
    axs[3].set_title('Ruido')

    # Ajustar layout y mostrar gráfico
    plt.tight_layout()
    plt.suptitle(f"Descomposición para {target_col}", fontsize=16, y=1.02)
    plt.show()
