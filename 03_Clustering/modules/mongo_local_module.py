import os
import pymongo
import pandas as pd
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def get_mongo_client(uri):
    """Conectar a MongoDB usando la URI proporcionada."""
    return pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)

def get_database(client, db_name):
    """Obtener la base de datos a partir del cliente MongoDB y nombre de la base de datos."""
    # Prueba la conexión
    client.server_info()  # Esto lanzará una excepción si no puede conectarse
    print("Conexión exitosa")
    return client[db_name]

def connect_to_db(db_name, local=True):
    """Establecer la conexión a MongoDB y obtener la base de datos."""
    if local:
        uri_var = 'LOCAL.URI'
    
    # Obtener la URI de MongoDB desde las variables de entorno
    uri = os.getenv(uri_var)  # Asegúrate de que esta variable esté en tu archivo .env

    if not uri:
        raise ValueError("No se ha encontrado la URI de MongoDB en las variables de entorno")

    client = get_mongo_client(uri)
    try:
        # Prueba la conexión
        client.server_info()
    except Exception as e:
        raise RuntimeError(f"Error de conexión: {e}")
    db = get_database(client, db_name)
    return db

def get_collection(db, collection_name):
    """Obtener una colección específica de la base de datos."""
    return db[collection_name]

def count_documents(collection):
    """Contar el número de documentos en una colección."""
    return collection.count_documents({})

def fetch_documents(collection):
    """Obtener todos los documentos de una colección y convertirlos a una lista de diccionarios."""
    return list(collection.find())

def convert_to_dataframe(documents):
    """Convertir una lista de documentos a un DataFrame de pandas."""
    return pd.DataFrame(documents)

def get_collection_as_dataframe(db, collection_name):
    """Obtener una colección como DataFrame."""
    collection = get_collection(db, collection_name)
    document_count = count_documents(collection)
    print(f"El número de documentos en la colección '{collection_name}' es: {document_count}")
    documents = fetch_documents(collection)
    return convert_to_dataframe(documents)
