import featuretools as ft
import re

def create_features(df_items, df_shops, df_ws, fc_config):
    # Crear el EntitySet
    es = ft.EntitySet(id="sales_data")

    # Añadir las tablas como dataframes
    es = es.add_dataframe(dataframe_name="items", dataframe=df_items, index="item")
    es = es.add_dataframe(dataframe_name="shops", dataframe=df_shops, index="store_code")
    es = es.add_dataframe(dataframe_name="sales", dataframe=df_ws, 
                          index="id", make_index=True, time_index="week") # Hay que cambiar time index a day??

    # Definir las relaciones entre tablas
    es = es.add_relationship("shops", "store_code", "sales", "store_code")
    es = es.add_relationship("items", "item", "sales", "item")

    # Configuramos creacion caracteristicas
    target_df = fc_config['target_df']
    agg_primitives = fc_config['agg_primitives']
    trans_primitives = fc_config['trans_primitives']
    max_depth = fc_config['max_depth']

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name=target_df,
        agg_primitives=agg_primitives,  
        trans_primitives=trans_primitives,
        max_depth=max_depth
    )

    # # Imprimir descripciones de caracteristicas creadas
    # get_features_descriptions(feature_defs)

    return feature_matrix, feature_defs

# Obtener descripciones de las características
def get_features_descriptions(features):
    descriptions = []  # Inicializar lista vacía
    for desc in features:
        feat_desc = ft.describe_feature(desc)
        descriptions.append(f'{desc}: {feat_desc}')  # Agregar a la lista
        # print(f'{desc}: {feat_desc}')
    return descriptions

# Seleccionar caracteristicas
def select_features(feature_matrix, selected_features):
    fm_selected = feature_matrix[list(selected_features.keys())]
    return fm_selected

# Renombrar caracteristicas
def rename_features(feature_matrix, selected_features):
    feature_matrix = feature_matrix.rename(columns=selected_features)
    descriptions = (
        feature_matrix.columns.to_series()
        .apply(lambda x: x.replace('_', ' ').title())
    )
    feature_matrix.columns = descriptions
    return feature_matrix

def select_features(features, substrings):
    # Subcadenas que quieres buscar
    # substrings = ['(sales.', '(raw_earn)', '.sell_price)']
    filtered_features = [feature for feature in features if any(sub in feature for sub in substrings)]

    filtered_features = extract_feature_name(filtered_features) # Nos quedamos solo con la parte del str que necesitamos

    return filtered_features

def extract_feature_name(features):
    extracted_features = []
    pattern = r'<Feature:\s*(.*?)>'  # Expresión regular para capturar el texto entre "<Feature:" y ">"
    
    for feature in features:
        match = re.search(pattern, feature)
        if match:
            extracted_features.append(match.group(1).strip())  # Capturamos la parte deseada

    return extracted_features

# Filtrar feature_matrix usando libreria featureTools
def filter_feature_matrix(feature_matrix):
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print(f'Shape before filtering: {feature_matrix.shape[1]}')
    filtered_features_matrix = ft.selection.remove_low_information_features(feature_matrix)
    filtered_features_matrix = ft.selection.remove_highly_correlated_features(filtered_features_matrix)
    filtered_features_matrix = ft.selection.remove_highly_null_features(filtered_features_matrix)
    filtered_features_matrix = ft.selection.remove_single_value_features(filtered_features_matrix)
    print(f'Shape after filtering: {filtered_features_matrix.shape[1]}')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    return filtered_features_matrix