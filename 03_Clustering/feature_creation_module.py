import featuretools as ft

def create_features(df_items, df_shops, df_ws):
    # Crear el EntitySet
    es = ft.EntitySet(id="sales_data")

    # Añadir las tablas como dataframes
    es = es.add_dataframe(dataframe_name="items", dataframe=df_items, index="item")
    es = es.add_dataframe(dataframe_name="shops", dataframe=df_shops, index="store_code")
    es = es.add_dataframe(dataframe_name="sales", dataframe=df_ws, 
                          index="id", make_index=True, time_index="week")

    # Definir las relaciones entre tablas
    es = es.add_relationship("shops", "store_code", "sales", "store_code")
    es = es.add_relationship("items", "item", "sales", "item")

    # Crear características agregadas relevantes para los productos
    feature_matrix_sales, feature_defs_sales = ft.dfs(
        entityset=es,
        target_dataframe_name="items",
        agg_primitives=["sum", "mean", "std", "count"],  
        trans_primitives=[],
        max_depth=2
    )

    get_features_descriptions(feature_defs_sales)

    # Seleccionar y renombrar características relevantes para el clustering
    selected_features = {
        'COUNT(sales)': 'num_sales',                   
        'SUM(sales.units)': 'total_units_sold',        
        'SUM(sales.raw_earn)': 'total_revenue',        
        'MEAN(sales.sell_price)': 'avg_sell_price',    
        'STD(sales.sell_price)': 'std_sell_price',     
        'STD(sales.units)': 'std_units_sold',          
        'MEAN(sales.units)': 'avg_units_per_week',     
        'MEAN(sales.week)': 'avg_week',                
        'STD(sales.week)': 'std_week',                 
        'MEAN(sales.year)': 'avg_year'                 
    }

    # Filtrar y renombrar la matriz de características
    fm_selected_sales = feature_matrix_sales[list(selected_features.keys())]
    fm_selected_sales.rename(columns=selected_features, inplace=True)

    return fm_selected_sales

# Obtener descripciones de las características
def get_features_descriptions(features):
    for desc in features:
        feat_desc = ft.describe_feature(desc)
        print(f'{desc}: {feat_desc}')

# Renombrar caracteristicas
def rename_features(feature_matrix):
    descriptions = (
        feature_matrix.columns.to_series()
        .apply(lambda x: x.replace('_', ' ').title())
    )
    feature_matrix.columns = descriptions
    return feature_matrix