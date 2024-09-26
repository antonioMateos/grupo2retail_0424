import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError

def preprocess_features(feature_matrix):
    try:
        # Identificar columnas categóricas y numéricas
        categorical_columns = feature_matrix.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_columns = feature_matrix.select_dtypes(include=[float, int]).columns.tolist()

        print(f'Cat cols: {len(categorical_columns)}')
        print(f'Num cols: {len(numeric_columns)}')

        # Definir la lista de transformadores
        transformers = []
        
        # Agregar el RobustScaler para las columnas numéricas si existen
        if numeric_columns:
            transformers.append(('num', RobustScaler(), numeric_columns))
        
        # Agregar el OneHotEncoder para las columnas categóricas si existen
        if categorical_columns:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns))

        # Crear el preprocesador con los transformadores existentes
        preprocessor = ColumnTransformer(transformers)

        # Aplicar las transformaciones utilizando el preprocesador
        scaled_features = preprocessor.fit_transform(feature_matrix)

        # Obtener el nombre de las columnas después de la transformación
        all_columns = numeric_columns.copy()  # Empezamos con las columnas numéricas

        if 'cat' in preprocessor.named_transformers_ and categorical_columns:
            encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
            all_columns += encoded_columns.tolist()

        # Convertir el resultado de nuevo a un DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=all_columns)

        # Devolver el DataFrame transformado
        return scaled_df
    
    except NotFittedError as e:
        print(f"Error: {e}")
        return None

# Uso
# scaled_df = preprocess_features(fm_selected_sales)