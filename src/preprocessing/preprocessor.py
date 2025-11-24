"""
Módulo que integra la normalización y discretización de datos
para el preprocesamiento de los conjuntos de datos.
"""

import pandas as pd
from datetime import datetime
from .normalizer import Normalizer
from .discretizer import Discretizer
from typing import Tuple, Dict

class Preprocessor:
    """
    Clase que integra los métodos de normalización y discretización de datos.
    
    Attributos:
        input_path (str): Ruta del archivo CSV de entrada.
        output_path (str): Ruta del directorio donde se guardarán los archivos procesados.
        df (pd.DataFrame): DataFrame que contiene los datos cargados desde el archivo CSV.
        normalizer (Normalizer): Instancia de la clase Normalizer para normalizar datos.
        discretizer (Discretizer): Instancia de la clase Discretizer para discretizar datos.
    """
    
    def __init__(self, input_path: str, output_path: str):
        """
        Inicializa el preprocesador cargando los datos y creando las composición de clases.
        
        Parámetros:
            input_path (str): Ruta del archivo CSV de entrada.
            output_path (str): Ruta del directorio de salida.
        """
        self.input_path = input_path
        self.output_path = output_path
        
        self.df = pd.read_csv(filepath_or_buffer=self.input_path, sep=',', header=None, index_col=False)
        
        self.normalizer = Normalizer()
        self.discretizer = Discretizer(self.df)
    
    def set_input_path(self, new_path: str):
        """
        Actualiza la ruta del archivo de entrada y recarga los datos en el DataFrame y en Discretizer.
        """
        self.input_path = new_path
        self.df = pd.read_csv(filepath_or_buffer=self.input_path, sep=',', header=None, index_col=False)
        
        self.discretizer.set_input_path(new_path)
        
    def summarize_data(self):
        """
        Muestra un resumen del número de filas y columnas del DataFrame.
        """
        print("\nResumen de los datos:")
        print(f"\tNúmero de filas: {self.df.shape[0]}")
        print(f"\tNúmero de columnas: {self.df.shape[1]}")
        
    def normalize(self, method: str, **kwargs) -> pd.DataFrame:
        """
        Aplica un algoritmo de normalización al DataFrame y devuelve el resultado.
        """
        if method == 'min_max':
            # Obtiene los límites inferior y superior
            lower_bound = kwargs.get('lower_bound', 0.0)
            upper_bound = kwargs.get('upper_bound', 10.0)
            return self.normalizer.min_max(self.df, lower_bound, upper_bound)
        elif method == 'z_score':
            return self.normalizer.z_score(self.df)
        else:
            return self.normalizer.decimal_scaling(self.df)
        
    def discretize(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Realiza la discretización basada en entropía usando cuartiles y devuelve las dos particiones y sus metadatos.
        """
        # Calcula la mejor división basada en ganancia de información
        self.discretizer.compute_info_gain()
        # Realiza la partición y obtiene los metadatos
        df_left, df_right, metadata = self.discretizer.discretize()
        
        return df_left, df_right, metadata
        
    def save_data(self, df: pd.DataFrame, method: str):
        """
        Guarda un DataFrame un archivo CSV.
        """
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.output_path}/{method}_{timestamp}.csv"
        df.to_csv(file_path, sep=',', header=False, index=False)
        
        print(f"Datos guardados en: {self.output_path}")
        
    def save_metadata(self, metadata: Dict):
        """
        Guarda los metadatos de la discretización en un archivo CSV.
        """
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.output_path}/metadata_{timestamp}.csv"
        
        # Convierte los metadatos en un DataFrame
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(file_path, sep=',', header=True, index=False)
        
        print(f"Metadatos guardados en: {self.output_path}")