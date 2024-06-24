import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

path = "data\main_clean.csv"

# Cargar los datos
@st.cache_data
def load_data():
    # Supongamos que los datos están en un archivo CSV
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

df = load_data()

# Evolución histórica del total de ventas
st.header('Evolución Histórica del Total de Ventas')

# Opciones de visualización
option = st.selectbox('Ver histórico total o por años', ['Total', 'Por años'])

if option == 'Total':
    # Total de ventas por año
    total_sales_per_year = df.groupby('year')['daily_sales'].sum().reset_index()

    # Producto más y menos vendido en total
    total_sales_by_product = df.groupby('product')['daily_sales'].sum()
    most_sold_product = total_sales_by_product.idxmax()
    most_sold_product_sales = total_sales_by_product.max()
    least_sold_product = total_sales_by_product.idxmin()
    least_sold_product_sales = total_sales_by_product.min()

    st.write(f'Producto más vendido: {most_sold_product} con {most_sold_product_sales} ventas')
    st.write(f'Producto menos vendido: {least_sold_product} con {least_sold_product_sales} ventas')
    
    # st.line_chart(total_sales) # -> Graf de linea
    # Crear el gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(total_sales_per_year['year'], total_sales_per_year['daily_sales'])
    ax.set_xlabel('Año')
    ax.set_ylabel('Total de Ventas')
    ax.set_title('Total de Ventas por Año (2011-2016)')
    st.pyplot(fig)
else:
    selected_year = st.selectbox('Seleccionar Año', sorted(df['year'].unique()))

    # Filtrar datos por el año seleccionado
    year_data = df[df['year'] == selected_year]

    # Producto más y menos vendido del año
    annual_sales_by_product = year_data.groupby('product')['daily_sales'].sum()
    most_sold_annual_product = annual_sales_by_product.idxmax()
    most_sold_annual_product_sales = annual_sales_by_product.max()
    least_sold_annual_product = annual_sales_by_product.idxmin()
    least_sold_annual_product_sales = annual_sales_by_product.min()

    st.write(f'Producto más vendido del año: {most_sold_annual_product} con {most_sold_annual_product_sales} ventas')
    st.write(f'Producto menos vendido del año: {least_sold_annual_product} con {least_sold_annual_product_sales} ventas')
    
    # Ventas mensuales del año seleccionado
    monthly_sales = year_data.groupby('month')['daily_sales'].sum().reset_index()
    st.header(f'Ventas Mensuales en {selected_year}')
    
    # Crear el gráfico de líneas para ventas mensuales
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['month'], monthly_sales['daily_sales'], marker='o')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Total de Ventas')
    ax.set_title(f'Total de Ventas por Mes en {selected_year}')
    st.pyplot(fig)
    
    # Análisis mensual detallado
    # for month in sorted(year_data['month'].unique()):
    #     st.subheader(f'Mes: {month}')
    #     monthly_data = year_data[year_data['month'] == month]
        
    #     # Ventas diarias del mes
    #     daily_sales = monthly_data.groupby('date')['daily_sales'].sum().reset_index()
        
    #     # Crear el gráfico de líneas para ventas diarias
    #     fig, ax = plt.subplots()
    #     ax.plot(daily_sales['date'], daily_sales['daily_sales'], marker='o')
    #     ax.set_xlabel('Fecha')
    #     ax.set_ylabel('Total de Ventas')
    #     ax.set_title(f'Total de Ventas Diarias en {selected_year}-{month}')
    #     st.pyplot(fig)